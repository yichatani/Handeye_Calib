import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tf
# 假设数据：A_i 和 B_i 是 4x4 齐次变换矩阵的集合
# A: [N, 4, 4], B: [N, 4, 4]

board2camera_matrix = np.load("/home/glab/Hardware_WS/src/handeye_calib/data/trial1/poses/board2camera_matrix.npy")
ee2base_matrix = np.load("/home/glab/Hardware_WS/src/handeye_calib/data/trial1/poses/ee2base_matrix.npy")

pose_num1 = len(board2camera_matrix)
pose_num2 = len(ee2base_matrix)
print("pose_num1:", pose_num1)
print("pose_num2: ", pose_num2)
# assert pose_num1 == pose_num2, "The number of poses in A and B should be the same"
pose_num = 15

# A = torch.tensor([...], dtype=torch.float32)  # 相机位姿
# B = torch.tensor([...], dtype=torch.float32)  # 机器人位姿




# 参数化 X：四元数 q (4维) 和平移 t (3维)
class HandEyeModel(nn.Module):
    def __init__(self, X):
        super(HandEyeModel, self).__init__()
        # 将输入的齐次矩阵转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to("cuda:0")
        
        # 从X中提取旋转矩阵和平移向量
        R = X_tensor[:3, :3]
        t = X_tensor[:3, 3]
        
        # 将旋转矩阵转换为四元数（需自定义函数）
        q = self.matrix_to_quaternion(R)
        
        # 初始化可训练参数
        self.q = nn.Parameter(q)  # 四元数
        self.t = nn.Parameter(t)  # 平移向量


    def quaternion_to_matrix(self, q):
        # 将四元数转换为旋转矩阵
        q = q / torch.norm(q)  # 归一化
        w, x, y, z = q
        R = torch.tensor([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ], dtype=torch.float32).to("cuda:0")
        return R
    
    def matrix_to_quaternion(self, R):
        # 将旋转矩阵转换为四元数
        tr = torch.trace(R)
        if tr > 0:
            S = torch.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        return torch.tensor([w, x, y, z], dtype=torch.float32).to("cuda:0")
    
    def forward(self):
        # 构造变换矩阵 X
        R = self.quaternion_to_matrix(self.q)
        T = torch.eye(4, device="cuda:0")
        T[:3, :3] = R
        T[:3, 3] = self.t
        return T

# 损失函数
def compute_loss(model, A, B):
    X = model()
    loss = 0
    for i in range(len(A)):
        AX = A[i] @ X
        XB = X @ B[i]
        loss += torch.norm(AX - XB, p='fro')**2
    return loss


if __name__ == "__main__":
    As = []
    Bs = []
    X = np.eye(4)
    xyzrpy = [0.03588,-0.08959,0.11999, 0, -0.0095, -0.0042]
    X[:3, :3] = tf.transformations.euler_matrix(xyzrpy[3], xyzrpy[4], xyzrpy[5])[:3, :3]
    X[:3, 3] = xyzrpy[:3]
    print("X:",X)
    # print("xyzrpy[:3]",xyzrpy)
    loss_sum = 0
    for i in range(pose_num):
        for j in range(pose_num):
            if i != j:
                A = np.linalg.inv(ee2base_matrix[j]) @ ee2base_matrix[i]
                B = board2camera_matrix[j] @ np.linalg.inv(board2camera_matrix[i])
                loss = (A @ X - X @ B )**2
                loss = np.linalg.norm(loss)
                loss_sum += loss
                # print("loss:",loss)
                As.append(A)
                Bs.append(B)
    
    print("loss_sum:",loss_sum)


    #### check matrixs
    for i in range(pose_num):
        T_ee2base = ee2base_matrix[i]
        T_board2camera = board2camera_matrix[i]
        T_camera2ee = X
        T_board2base = T_ee2base @ T_camera2ee @ T_board2camera
        # print("*******************")
        # print("T_ee2base:\n", T_ee2base)
        # print("T_board2camera:\n", T_board2camera)
        # print("T_board2base:\n", T_board2base)
        # print("*******************")

    As = np.array(As)
    Bs = np.array(Bs)
    A = torch.tensor(As, dtype=torch.float32).to("cuda:0")
    B = torch.tensor(Bs, dtype=torch.float32).to("cuda:0")
    model = HandEyeModel(X).to("cuda:0")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100000):
        optimizer.zero_grad()
        loss = compute_loss(model, A, B)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            T_camera2ee = model().cpu().detach().numpy()
            print("Hand-Eye Transform Matrix:\n", T_camera2ee)
            rpy = tf.transformations.euler_from_matrix(T_camera2ee)
            xyz = T_camera2ee[:3, 3]
            print("xyz:",xyz)
            print("rpy:",rpy)

    

    

    
                