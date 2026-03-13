import numpy as np
import glob
from pathlib import Path
import kitti_metric
import os


def load_kiss_fps(file):
    metrics_file = Path(file).parent.joinpath("result_metrics.log")
    with open(metrics_file,"r") as f:
        lines = f.readlines()
        for line in lines:
            if "Average Frequency" in line:
                fps = line.split("|")[2]
                return round(float(fps))
    
def load_ours_fps(file):
    metrics_file = file.replace("txt","log")
    with open(metrics_file,"r") as f:
        lines = f.readlines()
        fps_line = lines[-8]
        fps = fps_line.split(":")[-1].replace("frame/s","")
        return round(float(fps))
    
def load_mad_fps(file):
    metrics_file = file.replace("estimate.txt","timings.log")
    with open(metrics_file,"r") as f:
        lines = f.readlines()
        needed_time = float(lines[-1])
        fps = len(lines) / needed_time
        return round(float(fps))


def apply_kitti_calibration(home_dir,seq, poses: np.ndarray) -> np.ndarray:
    """Converts from Velodyne to Camera Frame"""
    import pykitti
    drive = pykitti.odometry(home_dir, seq)
    T_cam0_velo = drive.calib.T_cam0_velo
    return np.linalg.inv(T_cam0_velo) @ poses @ T_cam0_velo

def load_kitti_gt_poses(home_dir,seq) -> np.ndarray:
    """Converts from Velodyne to Camera Frame"""
    import pykitti
    drive = pykitti.odometry(home_dir, seq)
    T_cam0_velo = drive.calib.T_cam0_velo
    poses = np.array(drive.poses)
    poses = np.linalg.inv(T_cam0_velo) @ poses @ T_cam0_velo
    #poses[:,:3,:3] = Rotation.from_matrix(poses[:,:3,:3]).as_matrix()
    return poses


def get_mulran_calibration():
    # Apply calibration obtainbed from calib_base2ouster.txt
    #  T_lidar_to_base[:3, 3] = np.array([1.7042, -0.021, 1.8047])
    #  T_lidar_to_base[:3, :3] = tu_vieja.from_euler(
    #  "xyz", [0.0001, 0.0003, 179.6654], degrees=True
    #  )
    T_lidar_to_base = np.array(
        [
            [-9.9998295e-01, -5.8398386e-03, -5.2257060e-06, 1.7042000e00],
            [5.8398386e-03, -9.9998295e-01, 1.7758769e-06, -2.1000000e-02],
            [-5.2359878e-06, 1.7453292e-06, 1.0000000e00, 1.8047000e00],
            [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
        ]
    )
    T_base_to_lidar = np.linalg.inv(T_lidar_to_base)
    return T_lidar_to_base, T_base_to_lidar

def apply_mulran_calibration(home_dir,seq,poses):
    T_lidar_to_base, T_base_to_lidar = get_mulran_calibration()
    return T_lidar_to_base @ poses @ T_base_to_lidar

# Taket from KISS-ICP MulRan dataloader
def load_mulran_gt_poses(home_dir, seq):
    poses_file = str(Path(home_dir).joinpath(seq).joinpath("global_pose.csv"))
    velodyne_dir = str(Path(home_dir).joinpath(seq).joinpath("Ouster"))
    scan_files = sorted(glob.glob(velodyne_dir + "/*.bin"))
    scan_timestamps = [int(os.path.basename(t).split(".")[0]) for t in scan_files]


    """MuRan has more poses than scans, therefore we need to match 1-1 timestamp with pose"""
    def read_csv(poses_file: str):
        poses = np.loadtxt(poses_file, delimiter=",")
        timestamps = poses[:, 0]
        poses = poses[:, 1:]
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)),
            axis=1,
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return poses, timestamps

    # Read the csv file
    poses, timestamps = read_csv(poses_file)
    # Extract only the poses that has a matching Ouster scan
    poses = poses[[np.argmin(abs(timestamps - t)) for t in scan_timestamps]]

    # Convert from global coordinate poses to local poses
    first_pose = poses[0, :, :]
    poses = np.linalg.inv(first_pose) @ poses

    return apply_mulran_calibration(None,None,poses)

def load_odyssey_gt_poses(home_dir,seq):
    file = Path(home_dir).joinpath(seq).joinpath("refsys").joinpath("lidar_poses.txt")
    return load_poses_kitti_format(file)

def load_poses_kitti_format(poses_file):
    _poses = np.loadtxt(poses_file, delimiter=" ").reshape((-1,3,4))
    poses = np.zeros((len(_poses),4,4))
    poses[:,:3] = _poses
    poses[:,-1,-1] = 1
    return poses


def run_evaluation_on_kitti():
    # -------------------------------------------
    # Evaluation on the KITTI-Odometry dataset
    # -------------------------------------------
    ours_base_dir = "./estimates/Ours/baseline/kitti"
    ours_files = np.array(glob.glob(ours_base_dir + "/*.txt"))
    ours_files.sort()

    kissicp_base_dir = "./estimates/KISS-ICP/kitti"
    kissicp_files = np.array(glob.glob(kissicp_base_dir + "/*/*_poses_kitti.txt"))
    keys = [Path(e).stem.replace("_poses_kitti","") for e in kissicp_files]
    kissicp_files = kissicp_files[np.argsort(keys)]

    kissslam_base_dir = "./estimates/KISS-SLAM/kitti"
    kissslam_files = np.array(glob.glob(kissslam_base_dir + "/*/*_poses_kitti.txt"))
    keys = [Path(e).stem.replace("_poses_kitti","") for e in kissslam_files]
    kissslam_files = kissslam_files[np.argsort(keys)]

    madicp_base_dir = "./estimates/MAD-ICP/kitti"
    madicp_files = np.array(glob.glob(madicp_base_dir + "/*/estimate.txt"))
    madicp_files.sort()

    ours_all_errors = np.zeros((0,11),np.float64)
    kissicp_all_errors = np.zeros((0,11),np.float64)
    kissslam_all_errors = np.zeros((0,11),np.float64)
    madicp_all_errors = np.zeros((0,11),np.float64)
    rpe_ours_all_errors = np.zeros((0,11),np.float64)
    rpe_kissicp_all_errors = np.zeros((0,11),np.float64)
    rpe_kissslam_all_errors = np.zeros((0,11),np.float64)
    rpe_madicp_all_errors = np.zeros((0,11),np.float64)

    lengths = [100]
    normalize = True
    remove_outliers = True

    ours_fps = []
    kissicp_fps = []
    kissslam_fps = []
    madicp_fps = []

    output_string = ""
    for ours_file, kissicp_file, kissslam_file, madicp_file in zip(ours_files,kissicp_files,kissslam_files,madicp_files):
        stem = str(Path(ours_file).stem)
        seq = stem.replace("poses_","")
        print(seq)

        gt_poses = load_kitti_gt_poses(kitti_base_dir,seq)
        ours_poses = load_poses_kitti_format(ours_file)
        kissicp_poses = apply_kitti_calibration(kitti_base_dir, seq, load_poses_kitti_format(kissicp_file))
        kissslam_poses = apply_kitti_calibration(kitti_base_dir, seq, load_poses_kitti_format(kissslam_file))
        madicp_poses = apply_kitti_calibration(kitti_base_dir, seq, load_poses_kitti_format(madicp_file))

        ours_seq_errors = kitti_metric.eval(gt_poses,ours_poses)
        ours_all_errors = np.concatenate((ours_all_errors,ours_seq_errors))
        ours_mean_seq_error = np.mean(ours_seq_errors,axis=0)
        rpe_ours_seq_errors = kitti_metric.eval(gt_poses,ours_poses,lengths=lengths,normalize=normalize)
        rpe_ours_all_errors = np.concatenate((rpe_ours_all_errors,rpe_ours_seq_errors))
        rpe_ours_mean_seq_error = np.mean(rpe_ours_seq_errors,axis=0)
        ours_fps.append(load_ours_fps(ours_file))

        kissicp_seq_errors = kitti_metric.eval(gt_poses,kissicp_poses)
        kissicp_all_errors = np.concatenate((kissicp_all_errors,kissicp_seq_errors))
        kissicp_mean_seq_error = np.mean(kissicp_seq_errors,axis=0)
        rpe_kissicp_seq_errors = kitti_metric.eval(gt_poses,kissicp_poses,lengths=lengths,normalize=normalize)
        rpe_kissicp_all_errors = np.concatenate((rpe_kissicp_all_errors,rpe_kissicp_seq_errors))
        rpe_kissicp_mean_seq_error = np.mean(rpe_kissicp_seq_errors,axis=0)
        kissicp_fps.append(load_kiss_fps(kissicp_file))

        kissslam_seq_errors = kitti_metric.eval(gt_poses,kissslam_poses)
        kissslam_all_errors = np.concatenate((kissslam_all_errors,kissslam_seq_errors))
        kissslam_mean_seq_error = np.mean(kissslam_seq_errors,axis=0)
        rpe_kissslam_seq_errors = kitti_metric.eval(gt_poses,kissslam_poses,lengths=lengths,normalize=normalize)
        rpe_kissslam_all_errors = np.concatenate((rpe_kissslam_all_errors,rpe_kissslam_seq_errors))
        rpe_kissslam_mean_seq_error = np.mean(rpe_kissslam_seq_errors,axis=0)
        kissslam_fps.append(load_kiss_fps(kissslam_file))

        madicp_seq_errors = kitti_metric.eval(gt_poses,madicp_poses)
        if remove_outliers and seq not in ["03"]:
            madicp_all_errors = np.concatenate((madicp_all_errors,madicp_seq_errors))
        madicp_mean_seq_error = np.mean(madicp_seq_errors,axis=0)
        rpe_madicp_seq_errors = kitti_metric.eval(gt_poses,madicp_poses,lengths=lengths,normalize=normalize)
        if remove_outliers and seq not in ["03"]:
            rpe_madicp_all_errors = np.concatenate((rpe_madicp_all_errors,rpe_madicp_seq_errors))
        rpe_madicp_mean_seq_error = np.mean(rpe_madicp_seq_errors,axis=0)
        madicp_fps.append(load_mad_fps(madicp_file))

        output_string += f"{seq} & {round(ours_mean_seq_error[4],2)} & {round(kissicp_mean_seq_error[4],2)} & {round(kissslam_mean_seq_error[4],2)} & {round(madicp_mean_seq_error[4],2)} & {round(rpe_ours_mean_seq_error[4],2)} & {round(rpe_kissicp_mean_seq_error[4],2)} & {round(rpe_kissslam_mean_seq_error[4],2)} & {round(rpe_madicp_mean_seq_error[4],2)} \\\\ \n"

    mean_ours_all_errors = np.mean(ours_all_errors,axis=0)
    mean_kissicp_all_errors = np.mean(kissicp_all_errors,axis=0)
    mean_kissslam_all_errors = np.mean(kissslam_all_errors,axis=0)
    mean_madicp_all_errors = np.mean(madicp_all_errors,axis=0)
    rpe_mean_ours_all_errors = np.mean(rpe_ours_all_errors,axis=0)
    rpe_mean_kissicp_all_errors = np.mean(rpe_kissicp_all_errors,axis=0)
    rpe_mean_kissslam_all_errors = np.mean(rpe_kissslam_all_errors,axis=0)
    rpe_mean_madicp_all_errors = np.mean(rpe_madicp_all_errors,axis=0)

    output_string += f"mean & {round(mean_ours_all_errors[4],2)} & {round(mean_kissicp_all_errors[4],2)} & {round(mean_kissslam_all_errors[4],2)} & {round(mean_madicp_all_errors[4],2)} & {round(rpe_mean_ours_all_errors[4],2)} & {round(rpe_mean_kissicp_all_errors[4],2)} & {round(rpe_mean_kissslam_all_errors[4],2)}  & {round(rpe_mean_madicp_all_errors[4],2)} \\\\ \n"
    print(output_string)
    print(round(np.mean(ours_fps),2),round(np.mean(kissicp_fps),2),round(np.mean(kissslam_fps),2),round(np.mean(madicp_fps),2))



def run_evaluation_on_mulran():
    # -------------------------------------------
    # Evaluation on the MulRan dataset
    # -------------------------------------------
    ours_base_dir = "./estimates/Ours/baseline/mulran"
    ours_files = np.array(glob.glob(ours_base_dir + "/*.txt"))
    ours_files.sort()

    kissicp_base_dir = "./estimates/KISS-ICP/mulran"
    kissicp_files = np.array(glob.glob(kissicp_base_dir + "/*/*_poses_kitti.txt"))
    keys = [Path(e).stem.replace("_poses_kitti","") for e in kissicp_files]
    kissicp_files = kissicp_files[np.argsort(keys)]

    kissslam_base_dir = "./estimates/KISS-SLAM/mulran"
    kissslam_files = np.array(glob.glob(kissslam_base_dir + "/*/*_poses_kitti.txt"))
    keys = [Path(e).stem.replace("_poses_kitti","") for e in kissslam_files]
    kissslam_files = kissslam_files[np.argsort(keys)]

    madicp_base_dir = "./estimates/MAD-ICP/mulran"
    madicp_files = np.array(glob.glob(madicp_base_dir + "/*/estimate.txt"))
    madicp_files.sort()

    ours_all_errors = np.zeros((0,11),np.float64)
    kissicp_all_errors = np.zeros((0,11),np.float64)
    kissslam_all_errors = np.zeros((0,11),np.float64)
    madicp_all_errors = np.zeros((0,11),np.float64)
    rpe_ours_all_errors = np.zeros((0,11),np.float64)
    rpe_kissicp_all_errors = np.zeros((0,11),np.float64)
    rpe_kissslam_all_errors = np.zeros((0,11),np.float64)
    rpe_madicp_all_errors = np.zeros((0,11),np.float64)

    lengths = [100]
    normalize = True

    ours_fps = []
    kissicp_fps = []
    kissslam_fps = []
    madicp_fps = []

    output_string = ""
    for ours_file, kissicp_file, kissslam_file, madicp_file in zip(ours_files,kissicp_files,kissslam_files,madicp_files):
        stem = str(Path(ours_file).stem)
        seq = stem.replace("poses_","")

        gt_poses = load_mulran_gt_poses(mulran_base_dir,seq)
        ours_poses = load_poses_kitti_format(ours_file)
        kissicp_poses = load_poses_kitti_format(kissicp_file)
        kissslam_poses = load_poses_kitti_format(kissslam_file)
        madicp_poses = apply_mulran_calibration(mulran_base_dir,seq,load_poses_kitti_format(madicp_file))

        ours_seq_errors = kitti_metric.eval(gt_poses,ours_poses)
        ours_all_errors = np.concatenate((ours_all_errors,ours_seq_errors))
        ours_mean_seq_error = np.mean(ours_seq_errors,axis=0)
        rpe_ours_seq_errors = kitti_metric.eval(gt_poses,ours_poses,lengths=lengths,normalize=normalize)
        rpe_ours_all_errors = np.concatenate((rpe_ours_all_errors,rpe_ours_seq_errors))
        rpe_ours_mean_seq_error = np.mean(rpe_ours_seq_errors,axis=0)
        ours_fps.append(load_ours_fps(ours_file))

        kissicp_seq_errors = kitti_metric.eval(gt_poses,kissicp_poses)
        kissicp_all_errors = np.concatenate((kissicp_all_errors,kissicp_seq_errors))
        kissicp_mean_seq_error = np.mean(kissicp_seq_errors,axis=0)
        rpe_kissicp_seq_errors = kitti_metric.eval(gt_poses,kissicp_poses,lengths=lengths,normalize=normalize)
        rpe_kissicp_all_errors = np.concatenate((rpe_kissicp_all_errors,rpe_kissicp_seq_errors))
        rpe_kissicp_mean_seq_error = np.mean(rpe_kissicp_seq_errors,axis=0)
        kissicp_fps.append(load_kiss_fps(kissicp_file))

        kissslam_seq_errors = kitti_metric.eval(gt_poses,kissslam_poses)
        kissslam_all_errors = np.concatenate((kissslam_all_errors,kissslam_seq_errors))
        kissslam_mean_seq_error = np.mean(kissslam_seq_errors,axis=0)
        rpe_kissslam_seq_errors = kitti_metric.eval(gt_poses,kissslam_poses,lengths=lengths,normalize=normalize)
        rpe_kissslam_all_errors = np.concatenate((rpe_kissslam_all_errors,rpe_kissslam_seq_errors))
        rpe_kissslam_mean_seq_error = np.mean(rpe_kissslam_seq_errors,axis=0)
        kissslam_fps.append(load_kiss_fps(kissslam_file))

        madicp_seq_errors = kitti_metric.eval(gt_poses,madicp_poses)
        madicp_all_errors = np.concatenate((madicp_all_errors,madicp_seq_errors))
        madicp_mean_seq_error = np.mean(madicp_seq_errors,axis=0)
        rpe_madicp_seq_errors = kitti_metric.eval(gt_poses,madicp_poses,lengths=lengths,normalize=normalize)
        rpe_madicp_all_errors = np.concatenate((rpe_madicp_all_errors,rpe_madicp_seq_errors))
        rpe_madicp_mean_seq_error = np.mean(rpe_madicp_seq_errors,axis=0)
        madicp_fps.append(load_mad_fps(madicp_file))

        output_string += f"{seq} & {round(ours_mean_seq_error[4],2)} & {round(kissicp_mean_seq_error[4],2)} & {round(kissslam_mean_seq_error[4],2)} & {round(madicp_mean_seq_error[4],2)} & {round(rpe_ours_mean_seq_error[4],2)} & {round(rpe_kissicp_mean_seq_error[4],2)} & {round(rpe_kissslam_mean_seq_error[4],2)} & {round(rpe_madicp_mean_seq_error[4],2)} \\\\ \n"

    mean_ours_all_errors = np.mean(ours_all_errors,axis=0)
    mean_kissicp_all_errors = np.mean(kissicp_all_errors,axis=0)
    mean_kissslam_all_errors = np.mean(kissslam_all_errors,axis=0)
    mean_madicp_all_errors = np.mean(madicp_all_errors,axis=0)
    rpe_mean_ours_all_errors = np.mean(rpe_ours_all_errors,axis=0)
    rpe_mean_kissicp_all_errors = np.mean(rpe_kissicp_all_errors,axis=0)
    rpe_mean_kissslam_all_errors = np.mean(rpe_kissslam_all_errors,axis=0)
    rpe_mean_madicp_all_errors = np.mean(rpe_madicp_all_errors,axis=0)

    output_string += f"mean & {round(mean_ours_all_errors[4],2)} & {round(mean_kissicp_all_errors[4],2)} & {round(mean_kissslam_all_errors[4],2)} & {round(mean_madicp_all_errors[4],2)} & {round(rpe_mean_ours_all_errors[4],2)} & {round(rpe_mean_kissicp_all_errors[4],2)} & {round(rpe_mean_kissslam_all_errors[4],2)}  & {round(rpe_mean_madicp_all_errors[4],2)} \\\\ \n"
    print(output_string)
    print(round(np.mean(ours_fps),2),round(np.mean(kissicp_fps),2),round(np.mean(kissslam_fps),2),round(np.mean(madicp_fps),2))


def run_evaluation_on_odyssey():
    # -------------------------------------------
    # Evaluation on the Odyssey dataset
    # -------------------------------------------
    ours_base_dir = "./estimates/Ours/baseline/odyssey"
    ours_files = np.array(glob.glob(ours_base_dir + "/*.txt"))
    ours_files.sort()

    kissicp_base_dir = "./estimates/KISS-ICP/odyssey"
    kissicp_files = np.array(glob.glob(kissicp_base_dir + "/*/*_poses_kitti.txt"))
    keys = [Path(e).stem.replace("_poses_kitti","") for e in kissicp_files]
    kissicp_files = kissicp_files[np.argsort(keys)]

    kissslam_base_dir = "./estimates/KISS-SLAM/odyssey"
    kissslam_files = np.array(glob.glob(kissslam_base_dir + "/*/*_poses_kitti.txt"))
    keys = [Path(e).stem.replace("_poses_kitti","") for e in kissslam_files]
    kissslam_files = kissslam_files[np.argsort(keys)]

    madicp_base_dir = "./estimates/MAD-ICP/odyssey"
    madicp_files = np.array(glob.glob(madicp_base_dir + "/*/estimate.txt"))
    madicp_files.sort()

    ours_all_errors = np.zeros((0,11),np.float64)
    kissicp_all_errors = np.zeros((0,11),np.float64)
    kissslam_all_errors = np.zeros((0,11),np.float64)
    madicp_all_errors = np.zeros((0,11),np.float64)
    rpe_ours_all_errors = np.zeros((0,11),np.float64)
    rpe_kissicp_all_errors = np.zeros((0,11),np.float64)
    rpe_kissslam_all_errors = np.zeros((0,11),np.float64)
    rpe_madicp_all_errors = np.zeros((0,11),np.float64)

    lengths = [100]
    normalize = True
    remove_outliers = True

    ours_fps = []
    kissicp_fps = []
    kissslam_fps = []
    madicp_fps = []

    output_string = ""
    for ours_file, kissicp_file, kissslam_file, madicp_file in zip(ours_files,kissicp_files,kissslam_files,madicp_files):
        stem = str(Path(ours_file).stem)
        seq = stem.replace("poses_","")

        if seq[:-1] == "Tunnel":
            continue

        gt_poses = load_odyssey_gt_poses(odyssey_base_dir,seq)
        ours_poses = load_poses_kitti_format(ours_file)
        kissicp_poses = load_poses_kitti_format(kissicp_file)
        kissslam_poses = load_poses_kitti_format(kissslam_file)
        madicp_poses = load_poses_kitti_format(madicp_file)

        ours_seq_errors = kitti_metric.eval(gt_poses,ours_poses)
        ours_all_errors = np.concatenate((ours_all_errors,ours_seq_errors))
        ours_mean_seq_error = np.mean(ours_seq_errors,axis=0)
        rpe_ours_seq_errors = kitti_metric.eval(gt_poses,ours_poses,lengths=lengths,normalize=normalize)
        rpe_ours_all_errors = np.concatenate((rpe_ours_all_errors,rpe_ours_seq_errors))
        rpe_ours_mean_seq_error = np.mean(rpe_ours_seq_errors,axis=0)
        ours_fps.append(load_ours_fps(ours_file))

        kissicp_seq_errors = kitti_metric.eval(gt_poses,kissicp_poses)
        if remove_outliers and seq not in ["HighwayTunnel3"]:
            kissicp_all_errors = np.concatenate((kissicp_all_errors,kissicp_seq_errors))
        kissicp_mean_seq_error = np.mean(kissicp_seq_errors,axis=0)
        rpe_kissicp_seq_errors = kitti_metric.eval(gt_poses,kissicp_poses,lengths=lengths,normalize=normalize)
        if remove_outliers and seq not in ["HighwayTunnel3"]:
            rpe_kissicp_all_errors = np.concatenate((rpe_kissicp_all_errors,rpe_kissicp_seq_errors))
        rpe_kissicp_mean_seq_error = np.mean(rpe_kissicp_seq_errors,axis=0)
        kissicp_fps.append(load_kiss_fps(kissicp_file))

        kissslam_seq_errors = kitti_metric.eval(gt_poses,kissslam_poses)
        if remove_outliers and seq not in ["HighwayTunnel3"]:
            kissslam_all_errors = np.concatenate((kissslam_all_errors,kissslam_seq_errors))
        kissslam_mean_seq_error = np.mean(kissslam_seq_errors,axis=0)
        rpe_kissslam_seq_errors = kitti_metric.eval(gt_poses,kissslam_poses,lengths=lengths,normalize=normalize)
        if remove_outliers and seq not in ["HighwayTunnel3"]:
            rpe_kissslam_all_errors = np.concatenate((rpe_kissslam_all_errors,rpe_kissslam_seq_errors))
        rpe_kissslam_mean_seq_error = np.mean(rpe_kissslam_seq_errors,axis=0)
        kissslam_fps.append(load_kiss_fps(kissslam_file))

        madicp_seq_errors = kitti_metric.eval(gt_poses,madicp_poses)
        madicp_all_errors = np.concatenate((madicp_all_errors,madicp_seq_errors))
        madicp_mean_seq_error = np.mean(madicp_seq_errors,axis=0)
        rpe_madicp_seq_errors = kitti_metric.eval(gt_poses,madicp_poses,lengths=lengths,normalize=normalize)
        rpe_madicp_all_errors = np.concatenate((rpe_madicp_all_errors,rpe_madicp_seq_errors))
        rpe_madicp_mean_seq_error = np.mean(rpe_madicp_seq_errors,axis=0)
        madicp_fps.append(load_mad_fps(madicp_file))

        output_string += f"{seq} & {round(ours_mean_seq_error[4],2)} & {round(kissicp_mean_seq_error[4],2)} & {round(kissslam_mean_seq_error[4],2)} & {round(madicp_mean_seq_error[4],2)} & {round(rpe_ours_mean_seq_error[4],2)} & {round(rpe_kissicp_mean_seq_error[4],2)} & {round(rpe_kissslam_mean_seq_error[4],2)} & {round(rpe_madicp_mean_seq_error[4],2)} \\\\ \n"

    mean_ours_all_errors = np.mean(ours_all_errors,axis=0)
    mean_kissicp_all_errors = np.mean(kissicp_all_errors,axis=0)
    mean_kissslam_all_errors = np.mean(kissslam_all_errors,axis=0)
    mean_madicp_all_errors = np.mean(madicp_all_errors,axis=0)
    rpe_mean_ours_all_errors = np.mean(rpe_ours_all_errors,axis=0)
    rpe_mean_kissicp_all_errors = np.mean(rpe_kissicp_all_errors,axis=0)
    rpe_mean_kissslam_all_errors = np.mean(rpe_kissslam_all_errors,axis=0)
    rpe_mean_madicp_all_errors = np.mean(rpe_madicp_all_errors,axis=0)

    output_string += f"mean & {round(mean_ours_all_errors[4],2)} & {round(mean_kissicp_all_errors[4],2)} & {round(mean_kissslam_all_errors[4],2)} & {round(mean_madicp_all_errors[4],2)} & {round(rpe_mean_ours_all_errors[4],2)} & {round(rpe_mean_kissicp_all_errors[4],2)} & {round(rpe_mean_kissslam_all_errors[4],2)}  & {round(rpe_mean_madicp_all_errors[4],2)} \\\\ \n"
    print(output_string)
    print(round(np.mean(ours_fps),2),round(np.mean(kissicp_fps),2),round(np.mean(kissslam_fps),2),round(np.mean(madicp_fps),2))


def run_ablation():
    # -------------------------------------------
    # Ablation study
    # -------------------------------------------
    ours_base_dir = "./estimates/Ours/baseline/mulran"
    ours_files = np.array(glob.glob(ours_base_dir + "/*.txt"))
    ours_files.sort()

    fullcov_ours_base_dir = "./estimates/Ours/infofull/mulran"
    fullcov_ours_files = np.array(glob.glob(fullcov_ours_base_dir + "/*.txt"))
    fullcov_ours_files.sort()

    eyecov_ours_base_dir = "./estimates/Ours/infoeye/mulran"
    eyecov_ours_files = np.array(glob.glob(eyecov_ours_base_dir + "/*.txt"))
    eyecov_ours_files.sort()

    gono_ours_base_dir = "./estimates/Ours/gono/mulran"
    gono_ours_files = np.array(glob.glob(gono_ours_base_dir + "/*.txt"))
    gono_ours_files.sort()

    golast_ours_base_dir = "./estimates/Ours/golast/mulran"
    golast_ours_files = np.array(glob.glob(golast_ours_base_dir + "/*.txt"))
    golast_ours_files.sort()

    singlekf_ours_base_dir = "./estimates/Ours/single_const/mulran"
    singlekf_ours_files = np.array(glob.glob(singlekf_ours_base_dir + "/*.txt"))
    singlekf_ours_files.sort()

    ours_all_errors = np.zeros((0,11),np.float64)
    fullcov_all_errors = np.zeros((0,11),np.float64)
    eyecov_all_errors = np.zeros((0,11),np.float64)
    gono_all_errors = np.zeros((0,11),np.float64)
    golast_all_errors = np.zeros((0,11),np.float64)
    singlekf_all_errors = np.zeros((0,11),np.float64)

    rpe_ours_all_errors = np.zeros((0,11),np.float64)
    rpe_fullcov_all_errors = np.zeros((0,11),np.float64)
    rpe_eyecov_all_errors = np.zeros((0,11),np.float64)
    rpe_gono_all_errors = np.zeros((0,11),np.float64)
    rpe_golast_all_errors = np.zeros((0,11),np.float64)
    rpe_singlekf_all_errors = np.zeros((0,11),np.float64)

    lengths = [100]
    normalize=True

    output_string = ""
    for ours_file, fullcov_ours_file, eyecov_ours_file, gono_ours_file, golast_ours_file, singlekf_ours_file in zip(ours_files,fullcov_ours_files,eyecov_ours_files,gono_ours_files,golast_ours_files,singlekf_ours_files):
        stem = str(Path(ours_file).stem)
        seq = stem.replace("poses_","")

        gt_poses = load_mulran_gt_poses(mulran_base_dir,seq)
        ours_poses = load_poses_kitti_format(ours_file)
        fullcov_poses = load_poses_kitti_format(fullcov_ours_file)
        eyecov_poses = load_poses_kitti_format(eyecov_ours_file)
        gono_poses = load_poses_kitti_format(gono_ours_file)
        golast_poses = load_poses_kitti_format(golast_ours_file)
        singlekf_poses = load_poses_kitti_format(singlekf_ours_file)

        ours_seq_errors = kitti_metric.eval(gt_poses,ours_poses)
        ours_all_errors = np.concatenate((ours_all_errors,ours_seq_errors))
        ours_mean_seq_error = np.mean(ours_seq_errors,axis=0)
        rpe_ours_seq_errors = kitti_metric.eval(gt_poses,ours_poses,lengths=lengths,normalize=normalize)
        rpe_ours_all_errors = np.concatenate((rpe_ours_all_errors,rpe_ours_seq_errors))
        rpe_ours_mean_seq_error = np.mean(rpe_ours_seq_errors,axis=0)

        fullcov_seq_errors = kitti_metric.eval(gt_poses, fullcov_poses)
        fullcov_all_errors = np.concatenate((fullcov_all_errors,fullcov_seq_errors))
        fullcov_mean_seq_error = np.mean(fullcov_seq_errors,axis=0)
        rpe_fullcov_seq_errors = kitti_metric.eval(gt_poses, fullcov_poses,lengths=lengths,normalize=normalize)
        rpe_fullcov_all_errors = np.concatenate((rpe_fullcov_all_errors,rpe_fullcov_seq_errors))
        rpe_fullcov_mean_seq_error = np.mean(rpe_fullcov_seq_errors,axis=0)

        eyecov_seq_errors = kitti_metric.eval(gt_poses,eyecov_poses)
        eyecov_all_errors = np.concatenate((eyecov_all_errors,eyecov_seq_errors))
        eyecov_mean_seq_error = np.mean(eyecov_seq_errors,axis=0)
        rpe_eyecov_seq_errors = kitti_metric.eval(gt_poses,eyecov_poses,lengths=lengths,normalize=normalize)
        rpe_eyecov_all_errors = np.concatenate((rpe_eyecov_all_errors,rpe_eyecov_seq_errors))
        rpe_eyecov_mean_seq_error = np.mean(rpe_eyecov_seq_errors,axis=0)

        gono_seq_errors = kitti_metric.eval(gt_poses, gono_poses)
        gono_all_errors = np.concatenate((gono_all_errors,gono_seq_errors))
        gono_mean_seq_error = np.mean(gono_seq_errors,axis=0)
        rpe_gono_seq_errors = kitti_metric.eval(gt_poses, gono_poses,lengths=lengths,normalize=normalize)
        rpe_gono_all_errors = np.concatenate((rpe_gono_all_errors,rpe_gono_seq_errors))
        rpe_gono_mean_seq_error = np.mean(rpe_gono_seq_errors,axis=0)

        golast_seq_errors = kitti_metric.eval(gt_poses, golast_poses)
        golast_all_errors = np.concatenate((golast_all_errors,golast_seq_errors))
        golast_mean_seq_error = np.mean(golast_seq_errors,axis=0)
        rpe_golast_seq_errors = kitti_metric.eval(gt_poses, golast_poses,lengths=lengths,normalize=normalize)
        rpe_golast_all_errors = np.concatenate((rpe_golast_all_errors,rpe_golast_seq_errors))
        rpe_golast_mean_seq_error = np.mean(rpe_golast_seq_errors,axis=0)

        singlekf_seq_errors = kitti_metric.eval(gt_poses, singlekf_poses)
        singlekf_all_errors = np.concatenate((singlekf_all_errors,singlekf_seq_errors))
        singlekf_mean_seq_error = np.mean(singlekf_seq_errors,axis=0)
        rpe_singlekf_seq_errors = kitti_metric.eval(gt_poses, singlekf_poses,lengths=lengths,normalize=normalize)
        rpe_singlekf_all_errors = np.concatenate((rpe_singlekf_all_errors,rpe_singlekf_seq_errors))
        rpe_singlekf_mean_seq_error = np.mean(rpe_singlekf_seq_errors,axis=0)

        output_string += f"{seq} & {round(ours_mean_seq_error[4],2)} & {round(fullcov_mean_seq_error[4],2)} & {round(eyecov_mean_seq_error[4],2)} & {round(gono_mean_seq_error[4],2)} & {round(golast_mean_seq_error[4],2)} & {round(singlekf_mean_seq_error[4],2)} & \
            {round(rpe_ours_mean_seq_error[4],2)} & {round(rpe_fullcov_mean_seq_error[4],2)} & {round(rpe_eyecov_mean_seq_error[4],2)} & {round(rpe_gono_mean_seq_error[4],2)} & {round(rpe_golast_mean_seq_error[4],2)} & {round(rpe_singlekf_mean_seq_error[4],2)} \\\\ \n"

    mean_ours_all_errors = np.mean(ours_all_errors,axis=0)
    mean_fullcov_all_errors = np.mean(fullcov_all_errors,axis=0)
    mean_eyecov_all_errors = np.mean(eyecov_all_errors,axis=0)
    mean_gono_all_errors = np.mean(gono_all_errors,axis=0)
    mean_golast_all_errors = np.mean(golast_all_errors,axis=0)
    mean_singlekf_all_errors = np.mean(singlekf_all_errors,axis=0)

    rpe_mean_ours_all_errors = np.mean(rpe_ours_all_errors,axis=0)
    rpe_mean_fullcov_all_errors = np.mean(rpe_fullcov_all_errors,axis=0)
    rpe_mean_eyecov_all_errors = np.mean(rpe_eyecov_all_errors,axis=0)
    rpe_mean_gono_all_errors = np.mean(rpe_gono_all_errors,axis=0)
    rpe_mean_golast_all_errors = np.mean(rpe_golast_all_errors,axis=0)
    rpe_mean_singlekf_all_errors = np.mean(rpe_singlekf_all_errors,axis=0)

    output_string += f"mean & {round(mean_ours_all_errors[4],2)} & {round(mean_fullcov_all_errors[4],2)} & {round(mean_eyecov_all_errors[4],2)} & {round(mean_gono_all_errors[4],2)} & {round(mean_golast_all_errors[4],2)} & {round(mean_singlekf_all_errors[4],2)} & \
        {round(rpe_mean_ours_all_errors[4],2)} & {round(rpe_mean_fullcov_all_errors[4],2)} & {round(rpe_mean_eyecov_all_errors[4],2)} & {round(rpe_mean_gono_all_errors[4],2)} & {round(rpe_mean_golast_all_errors[4],2)} & {round(rpe_mean_singlekf_all_errors[4],2)} \\\\ \n"
    
    print(output_string)



if __name__ == "__main__":
    # Change these
    kitti_base_dir = "/path/to/KITTI-odometry"
    odyssey_base_dir = "/path/to/Odyssey"
    mulran_base_dir = "/path/to/MulRan"

    odyssey_base_dir = "/media/aaron/dataset-ssd/datasets/odyssey"
    kitti_base_dir = "/media/aaron/dataset-ssd/datasets/KITTI-odometry"
    mulran_base_dir = "/media/aaron/dataset-ssd/datasets/MulRan"

    run_evaluation_on_kitti()
    run_evaluation_on_mulran()
    run_evaluation_on_odyssey()
    run_ablation()
        

