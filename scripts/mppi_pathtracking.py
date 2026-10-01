import itertools
import math
import numpy as np
from typing import Tuple
from pathtracking_kbm import Vehicle, BackgroundRecorder

class MPPIControllerForPathTracking():
    def __init__(
            self,
            delta_t: float = 0.05,
            wheel_base: float = 2.5, # [m]
            max_steer_abs: float = 0.523, # [rad]
            max_accel_abs: float = 2.000, # [m/s^2]
            ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.1]]), 
            stage_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            visualize_optimal_traj = True,  # if True, optimal trajectory is visualized
            visualze_sampled_trajs = False, # if True, sampled trajectories are visualized
    ) -> None:
        """initialize mppi controller for path-tracking"""
        # mppi parameters
        self.dim_x = 4 # dimension of system state vector
        self.dim_u = 2 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi
        self.Sigma = sigma # deviation of noise
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

        # vehicle parameters
        self.delta_t = delta_t #[s]
        self.wheel_base = wheel_base#[m]
        self.max_steer_abs = max_steer_abs # [rad]
        self.max_accel_abs = max_accel_abs # [m/s^2]
        self.ref_path = ref_path
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.u_limit = np.array([self.max_steer_abs, self.max_accel_abs])

        # mppi variables
        self.u_prev = np.zeros((self.T, self.dim_u))

        # ref_path info
        self.prev_waypoints_idx = 0

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        """calculate optimal control input"""
        # load privious control input sequence
        u = self.u_prev

        # set initial x value from observation
        x0 = observed_x

        # get the waypoint closest to current vehicle position 
        self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)

        # sample noise
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u) # size is self.K x self.T

        # control input sequence with noise: exploitation samples first, exploration samples after
        n_exploit = math.ceil((1.0 - self.param_exploration) * self.K)
        v = epsilon.copy()
        v[:n_exploit] += u
        v = np.clip(v, -self.u_limit, self.u_limit)

        # rollout all K samples at once
        S = self.param_gamma * np.einsum('ti,ij,ktj->k', u, self.Sigma_inv, v)
        traj = np.zeros((self.K, self.T, self.dim_x))
        x = np.tile(x0, (self.K, 1))
        for t in range(self.T):
            x = self._F(x, v[:, t])
            traj[:, t] = x
            S += self._c(x)
        S += self._phi(x)

        # compute information theoretic weights for each sample
        w = self._compute_weights(S)

        # calculate w_k * epsilon_k
        w_epsilon = np.einsum('k,ktu->tu', w, epsilon)

        # apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)

        # update control input sequence
        u += w_epsilon

        # calculate optimal trajectory
        optimal_traj = np.zeros((self.T, self.dim_x))
        if self.visualize_optimal_traj:
            np.clip(u, -self.u_limit, self.u_limit, out=u)
            x = x0[None]
            for t in range(0, self.T): # loop for time step t = 0 ~ T-1
                x = self._F(x, u[t][None])
                optimal_traj[t] = x[0]

        # calculate sampled trajectories
        sampled_traj_list = traj if self.visualze_sampled_trajs else np.zeros((self.K, self.T, self.dim_x))

        # update privious control input sequence (shift 1 step to the left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # return optimal control input and input sequence
        return u[0], u, optimal_traj, sampled_traj_list

    def _calc_epsilon(self, sigma: np.ndarray, size_sample: int, size_time_step: int, size_dim_u: int) -> np.ndarray:
        """sample epsilon"""
        # check if sigma row size == sigma col size == size_dim_u and size_dim_u > 0
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u or size_dim_u < 1:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            raise ValueError

        # sample epsilon
        mu = np.zeros((size_dim_u)) # set average as a zero vector
        epsilon = np.random.multivariate_normal(mu, sigma, (size_sample, size_time_step))
        return epsilon

    def _c(self, x_t: np.ndarray) -> np.ndarray:
        """calculate stage cost for a batch of states, shape (K, dim_x) -> (K,)"""
        return self._state_cost(x_t, self.stage_cost_weight)

    def _phi(self, x_T: np.ndarray) -> np.ndarray:
        """calculate terminal cost for a batch of states, shape (K, dim_x) -> (K,)"""
        return self._state_cost(x_T, self.terminal_cost_weight)

    def _state_cost(self, x_t: np.ndarray, weight: np.ndarray) -> np.ndarray:
        ref = self._nearest_ref(x_t[:, :2])
        err = x_t - ref
        err[:, 2] = np.arctan2(np.sin(err[:, 2]), np.cos(err[:, 2])) # normalize yaw error to [-pi, pi]
        return (err**2) @ weight

    def _nearest_ref(self, xy: np.ndarray) -> np.ndarray:
        """reference [x, y, yaw, v] of the nearest waypoint for each point in xy, shape (K, 2) -> (K, 4)"""
        segment = self.ref_path[self._search_window()]
        d = ((xy[:, None, :] - segment[None, :, :2])**2).sum(axis=2)
        return segment[d.argmin(axis=1)]

    def _search_window(self) -> np.ndarray:
        """indices of the forward search range, wrapping around since the reference path is a closed loop"""
        SEARCH_IDX_LEN = 200 # [points] forward search range
        return (self.prev_waypoints_idx + np.arange(SEARCH_IDX_LEN)) % self.ref_path.shape[0]

    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False):
        """search the closest waypoint to the vehicle on the reference path"""
        idx = self._search_window()
        d = (x - self.ref_path[idx, 0])**2 + (y - self.ref_path[idx, 1])**2
        nearest_idx = int(idx[d.argmin()])

        # get reference values of the nearest waypoint
        ref_x = self.ref_path[nearest_idx,0]
        ref_y = self.ref_path[nearest_idx,1]
        ref_yaw = self.ref_path[nearest_idx,2]
        ref_v = self.ref_path[nearest_idx,3]

        # update nearest waypoint index if necessary
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx 

        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v

    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state of the vehicle, shapes (N, dim_x), (N, dim_u) -> (N, dim_x)"""
        # get previous state variables
        x, y, yaw, v = x_t.T
        steer, accel = v_t.T

        # prepare params
        l = self.wheel_base
        dt = self.delta_t

        # update state variables
        new_x = x + v * np.cos(yaw) * dt
        new_y = y + v * np.sin(yaw) * dt
        new_yaw = yaw + v / l * np.tan(steer) * dt
        new_v = v + accel * dt

        # return updated state
        x_t_plus_1 = np.stack([new_x, new_y, new_yaw, new_v], axis=1)
        return x_t_plus_1

    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""
        w = np.exp((-1.0/self.param_lambda) * (S - S.min()))
        return w / w.sum()

    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        Note: The original MPPI paper uses the Savitzky-Golay Filter for smoothing control inputs.
        """
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean


def run_simulation_mppi_pathtracking() -> None:
    """run simulation of pathtracking with MPPI controller"""
    print("[INFO] Start simulation of pathtracking with MPPI controller")

    # simulation settings
    delta_t = 0.05 # [sec]
    print(f"[INFO] delta_t : {delta_t:.2f}[s], running until Ctrl+C")

    # load the reference path
    ref_path = np.genfromtxt('./data/ovalpath.csv', delimiter=',', skip_header=1)

    # initialize a vehicle as a control target
    vehicle_params = dict(
        wheel_base=2.5,
        max_steer_abs=0.523, # [rad]
        max_accel_abs=2.000, # [m/s^2]
        ref_path = ref_path[:, 0:2], # ndarray, size is <num_of_waypoints x 2>
    )
    vehicle = Vehicle(visualize=False, **vehicle_params)
    recorder = BackgroundRecorder("mppi_pathtracking_demo.mp4", fps=1.0/delta_t, **vehicle_params) # ffmpeg is required to write mp4 file
    vehicle.reset(
        init_state = np.array([0.0, 1.0, 0.0, 0.0]), # [x[m], y[m], yaw[rad], v[m/s]]
    )

    # initialize a mppi controller for the vehicle
    mppi = MPPIControllerForPathTracking(
        delta_t = delta_t*2.0, # [s]
        wheel_base = 2.5, # [m]
        max_steer_abs = 0.523, # [rad]
        max_accel_abs = 2.000, # [m/s^2]
        ref_path = ref_path, # ndarray, size is <num_of_waypoints x 2>
        horizon_step_T = 20, # [steps]
        number_of_samples_K = 500, # [samples]
        param_exploration = 0.0,
        param_lambda = 100.0,
        param_alpha = 0.98,
        sigma = np.array([[0.075, 0.0], [0.0, 2.0]]),
        stage_cost_weight = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
        terminal_cost_weight = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
        visualze_sampled_trajs = True
    )

    # simulation loop
    try:
        for i in itertools.count():

            # get current state of vehicle
            current_state = vehicle.get_state()

            # calculate input force with MPPI
            optimal_input, optimal_input_sequence, optimal_traj, sampled_traj_list = mppi.calc_control_input(
                observed_x = current_state
            )

            # print current state and input force
            print(f"Time: {i*delta_t:>2.2f}[s], x={current_state[0]:>+3.3f}[m], y={current_state[1]:>+3.3f}[m], yaw={current_state[2]:>+3.3f}[rad], v={current_state[3]:>+3.3f}[m/s], steer={optimal_input[0]:>+6.2f}[rad], accel={optimal_input[1]:>+6.2f}[m/s]")

            # update states of vehicle
            vehicle.update(u=optimal_input, delta_t=delta_t, append_frame=False)
            recorder.add_frame(vehicle.get_state(), optimal_input[0], optimal_input[1], optimal_traj[:, 0:2], sampled_traj_list[:, :, 0:2])
    except KeyboardInterrupt:
        print("[INFO] Ctrl+C detected. Terminate simulation.")
    finally:
        print(f"[INFO] Waiting for ~{recorder.queue.qsize()} remaining frames to be written...")
        recorder.close()

if __name__ == "__main__":
    run_simulation_mppi_pathtracking()
