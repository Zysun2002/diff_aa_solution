import torch
import math
import numpy as np
import torch
import math
import ipdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def cal_smooth_loss(points):

    diff = points[:-2] - 2 * points[1:-1] + points[2:]
    return (diff ** 2).mean()

def cal_smooth_loss_distance(points):
    # points: tensor of shape (N, D)
    diff = points[:-2] - 2 * points[1:-1] + points[2:]  # second-order diff
    
    # distance (magnitude) of each diff vector
    dist = torch.norm(diff, dim=-1)
    
    # mean squared distance
    return (dist ** 2).mean()



def cal_straightness_loss(points, peak_angle=178, eps=1e-8):
    """
    Straightness loss using tanh: nearly constant before peak_angle,
    drops smoothly toward 180 degrees.
    """

    n = points.shape[0]
    if n < 3:
        return torch.tensor(0.0, dtype=torch.float32, device=points.device)

    # vectors: (N-2, 2)
    ba = points[:-2] - points[1:-1]
    bc = points[2:]  - points[1:-1]

    # normalize
    ba = ba / (ba.norm(dim=1, keepdim=True) + eps)
    bc = bc / (bc.norm(dim=1, keepdim=True) + eps)

    # cosine similarity
    cos_theta = (ba * bc).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    # peak in radians
    peak_rad = torch.deg2rad(torch.tensor(peak_angle, dtype=torch.float32, device=points.device))
    cos_peak = torch.cos(peak_rad)  # corresponds to peak_angle

    # scale factor to control how fast it drops after peak
    alpha = 10.0

    # tanh loss: stays near 0.1 before peak, drops after
    penalties = 0.1 *  torch.tanh(alpha * (cos_theta - cos_peak))

    return penalties.mean()


def cal_axis_align_loss(points, eps=1e-8):
    # Compute edges
    edges = points[1:] - points[:-1]
    edges_norm = edges / (edges.norm(dim=1, keepdim=True) + eps)

    # assuming edges_norm is [N, 2]
    x_dir = torch.tensor([1.0, 0.0], device=edges_norm.device)
    y_dir = torch.tensor([0.0, 1.0], device=edges_norm.device)

    # inner products
    dot_x = (edges_norm * x_dir).sum(dim=1)  # [N]
    dot_y = (edges_norm * y_dir).sum(dim=1)  # [N]

    # absolute values
    abs_dot_x = dot_x.abs()
    abs_dot_y = dot_y.abs()

    # take elementwise minimum
    min_val = torch.max(abs_dot_x, abs_dot_y)  # [N]
    
    alpha = 50.0

    # tanh loss: stays near 0.1 before peak, drops after
    penalties = -0.01 * torch.tanh(alpha * (min_val - 0.9998)) + 0.01


    return penalties.mean()



def cal_curvature_loss(points, return_w=False):
    """
    Smooth differentiable curvature loss over a sequential list of 2D points.

    Args:
        points: (N,2) tensor of 2D points (in order)
        return_w: if True, also return the per-quadruple weights w
    """
    # Hyperparameters (fixed inside function)
    l_cont   = 60   # continuity scaling factor
    b_infl   = 0.1  # inflection bias
    l_infl   = 90   # inflection offset
    sharpness = 20  # controls softness of transition

    n = points.shape[0]
    if n < 4:
        return (torch.tensor(0.0, dtype=torch.float32, device=points.device),
                torch.tensor([]) if return_w else None)

    # Consecutive quadruples
    pi, pj, pk, pl = points[:-3], points[1:-2], points[2:-1], points[3:]

    # Vectors
    vij = pj - pi
    vjk = pk - pj
    vkl = pl - pk

    # Normalize helper
    def normalize(v):
        return v / (v.norm(dim=1, keepdim=True) + 1e-6)

    vij_n = normalize(vij)
    vjk_n = normalize(vjk)
    vkl_n = normalize(vkl)

    # Angles (in degrees)
    cos1 = (vij_n * vjk_n).sum(dim=1).clamp(-0.9999, 0.9999)
    cos2 = (vjk_n * vkl_n).sum(dim=1).clamp(-0.9999, 0.9999)

    # 2D cross products 
    cross1 = vij[:, 0] * vjk[:, 1] - vij[:, 1] * vjk[:, 0]
    cross2 = vjk[:, 0] * vkl[:, 1] - vjk[:, 1] * vkl[:, 0]

    # Smooth weight in [0,1]
    sign_val = cross1 * cross2 + 1e-3
    w = torch.sigmoid(sharpness * sign_val)

    # Two candidate values
    same_side_val = torch.abs(cos1 - cos2) * 1
    inflection_val = - torch.min(cos1, cos2) * 1 + 5

    # Smooth interpolation
    val = w * same_side_val + (1 - w) * inflection_val

    if return_w:
        return 0.05 * val.mean(), w.detach().cpu().numpy()
    return 0.05 * val.mean()



def test_curvature_loss():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import math, torch
    # Fixed first two edges to form ~135° at p1
    p1 = torch.tensor([0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0])
    angle1 = math.radians(45.0)
    p0 = p1 + torch.tensor([math.cos(angle1), math.sin(angle1)])  # fixed first edge

    angles = list(range(0, 360, 1))
    losses, weights = [], []

    for ang in angles:
        ang_rad = math.radians(ang)
        p3 = p2 + torch.tensor([math.cos(ang_rad), math.sin(ang_rad)])
        pts = torch.stack([p0, p1, p2, p3], dim=0)
        loss, w = cal_curvature_loss(pts, return_w=True)
        losses.append(loss.item())
        weights.append(w[0])  # only one quadruple

    # Show only 8 polylines
    angles_show = range(0, 360, 45)

    # --- One figure with GridSpec ---
    fig = plt.figure(figsize=(14, 8))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 4, figure=fig)  # 3 rows × 4 cols

    # Big loss plot across top row
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_xlabel("Second angle (degrees)")
    ax1.set_ylabel("Curvature loss", color="tab:blue")
    ax1.plot(angles, losses, color="tab:blue", marker="o", markersize=2, label="Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(45))

    ax2 = ax1.twinx()
    ax2.set_ylabel("Weight w", color="tab:red")
    ax2.plot(angles, weights, color="tab:red", linestyle="--", label="w")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("Curvature loss & weight vs. second angle")

    # Grid of 8 polylines in rows 2–3
    for i, ang in enumerate(angles_show):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])

        ang_rad = math.radians(ang)
        p3 = p2 + torch.tensor([math.cos(ang_rad), math.sin(ang_rad)])
        pts = torch.stack([p0, p1, p2, p3], dim=0)

        x, y = pts[:, 0], pts[:, 1]
        ax.plot(x, y, "o-", linewidth=2)
        ax.set_title(f"{ang}°", fontsize=9)
        ax.axis("equal")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def test_straight_loss():
    def create_triangle(angle_deg, length=1.0):
        """
        Create 3 points (A, B, C) with angle at B = angle_deg.
        - angle=180 -> straight line
        - angle=0   -> folded
        """
        angle_rad = math.radians(angle_deg)

        # B is the vertex
        B = torch.tensor([0.0, 0.0])
        # A fixed to the left
        A = torch.tensor([-length, 0.0])
        # BA vector
        BA = A - B

        # Rotate BA by angle_deg to get BC
        rot = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad),  math.cos(angle_rad)]
        ])
        BC = torch.matmul(rot, BA)

        C = B + BC
        return torch.stack([A, B, C], dim=0)

    # Prepare data
    angles = range(0, 181)
    losses = []
    for ang in angles:
        pts = create_triangle(ang).float()
        loss = cal_straightness_loss(pts)
        losses.append(loss.item())

    # --- Create ONE big figure ---
    fig = plt.figure(figsize=(18, 10))

    # --- First subplot: loss vs angle ---
    ax1 = plt.subplot2grid((3, 7), (0, 0), colspan=7)  # top row, full width
    ax1.plot(angles, losses, marker="o")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Angle (degrees)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Straightness loss vs angle")
    ax1.grid(True)
    ax1.set_xticks(range(0, 181, 15))

    # --- Second subplot: polyline grid ---
    grid_angles = range(0, 181, 15)  # 0, 15, 30, ..., 180
    for i, ang in enumerate(grid_angles):
        row, col = divmod(i, 7)
        ax = plt.subplot2grid((3, 7), (row + 1, col))  # rows 1–2
        pts = create_triangle(ang).float()
        x, y = pts[:, 0], pts[:, 1]
        ax.plot(x, y, "o-", linewidth=2)
        ax.set_title(f"{ang}°", fontsize=10)
        ax.axis("equal")
        ax.axis("off")

    plt.suptitle("Straightness Loss + Polylines at Different Angles", fontsize=16)
    plt.tight_layout()
    plt.show()


def test_axis_align_loss():
    angles_deg = np.arange(0, 361, 1)  # full sweep
    losses = []

    for ang in angles_deg:
        rad = math.radians(ang)
        p0 = torch.tensor([0.0, 0.0])
        p1 = torch.tensor([math.cos(rad), math.sin(rad)])
        pts = torch.stack([p0, p1], dim=0)

        loss = cal_axis_align_loss(pts)
        losses.append(loss.item())

    # --- Figure 1: loss curve ---
    plt.figure(figsize=(12, 6))
    plt.plot(angles_deg, losses, marker='o', markersize=3)
    plt.xlabel("Edge angle (degrees)")
    plt.ylabel("Axis alignment loss")
    plt.title("Axis alignment loss vs edge angle")

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(45))
    ax.set_ylim(0, 1.0)
    plt.grid(True)

    # --- Figure 2: polyline samples ---
    plt.figure(figsize=(12, 12))
    angles_show = range(0, 360, 15)

    for i, ang in enumerate(angles_show, 1):
        rad = math.radians(ang)
        p0 = torch.tensor([0.0, 0.0])
        p1 = torch.tensor([math.cos(rad), math.sin(rad)])
        pts = torch.stack([p0, p1], dim=0)

        x, y = pts[:, 0], pts[:, 1]

        plt.subplot(4, 6, i)  # 25 subplots max → fits in 30 slots
        plt.plot(x, y, "o-", linewidth=2)
        plt.title(f"{ang}°", fontsize=9)
        plt.axis("equal")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    test_curvature_loss()