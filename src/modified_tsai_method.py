import numpy as np

def homogeneous_transform(R, t):
    """Create a 4x4 homogeneous transformation matrix from rotation R and translation t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def solve_ax_xb_translation(A_matrices, B_matrices, R_gripper2cam):
    """
    Solve hand-eye calibration for translation only, given known rotation R_gripper2cam.
    A_matrices: List of gripper-to-base homogeneous transforms (4x4).
    B_matrices: List of camera-to-target homogeneous transforms (4x4).
    R_gripper2cam: Known 3x3 rotation matrix from gripper to camera.
    Returns: t_gripper2cam (3x1 translation vector).
    """
    n = len(A_matrices)
    if n < 2:
        raise ValueError("At least two sets of transformations are required.")
    if len(B_matrices) != n:
        raise ValueError("Number of A and B matrices must match.")

    # Construct the system for translation: A * t_gripper2cam = b
    A = []
    b = []
    for i in range(n):
        for j in range(i + 1, n):
            # Extract rotations and translations
            R_Ai = A_matrices[i][:3, :3]
            t_Ai = A_matrices[i][:3, 3]
            R_Aj = A_matrices[j][:3, :3]
            t_Aj = A_matrices[j][:3, 3]
            R_Bi = B_matrices[i][:3, :3]
            t_Bi = B_matrices[i][:3, 3]
            R_Bj = B_matrices[j][:3, :3]
            t_Bj = B_matrices[j][:3, 3]

            # Equation: (R_Ai - I) * t_gripper2cam = R_gripper2cam * (t_Bj - t_Bi) - (t_Aj - t_Ai)
            A.append(R_Ai - np.eye(3))
            b.append(R_gripper2cam @ (t_Bj - t_Bi) - (t_Aj - t_Ai))

    A = np.vstack(A)
    b = np.hstack(b)

    # Solve the linear system using least squares
    t_gripper2cam, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return t_gripper2cam

def hand_eye_calibration_tsai(A_matrices, B_matrices, R_gripper2cam):
    """
    Perform hand-eye calibration (eye-on-hand) using Tsai's method for translation only.
    Returns the homogeneous transformation matrix from gripper to camera.
    """
    # Solve for translation
    t_gripper2cam = solve_ax_xb_translation(A_matrices, B_matrices, R_gripper2cam)

    # Construct the final hand-eye transformation matrix
    T_gripper2cam = homogeneous_transform(R_gripper2cam, t_gripper2cam)
    return T_gripper2cam



# Example usage
if __name__ == "__main__":
    # Example data: known rotation matrix (identity for simplicity)
    # TODO read from file
    R_gripper2cam = np.array([[ 9.99946056e-01  ,4.19998765e-03 ,-9.49977332e-03 ],\
                                [-4.19979813e-03 , 9.99991180e-01,  3.98992825e-05],\
                                [ 9.49985710e-03,  0.00000000e+00 , 9.99954875e-01]\
                                ])


    # Simulated transformation matrices (A: gripper-to-base, B: camera-to-target)
    # TODO read from file

    board2camera_matrix = np.load("/home/glab/Hardware_WS/src/handeye_calib/data/trial1/poses/board2camera_matrix.npy")
    ee2base_matrix = np.load("/home/glab/Hardware_WS/src/handeye_calib/data/trial1/poses/ee2base_matrix.npy")
    As = []
    Bs = []
    pose_num = 15
    for i in range(pose_num):
        for j in range(pose_num):
            if i != j:
                A = np.linalg.inv(ee2base_matrix[j]) @ ee2base_matrix[i]
                B = board2camera_matrix[j] @ np.linalg.inv(board2camera_matrix[i])
                As.append(A)
                Bs.append(B)
    A_matrices = np.array(As)
    B_matrices = np.array(Bs)



    # Perform calibration
    T_gripper2cam = hand_eye_calibration_tsai(A_matrices, B_matrices, R_gripper2cam)
    print("Hand-eye transformation matrix (T_gripper2cam):")
    print(T_gripper2cam)