from expoly.frames import Frame
from expoly.carve import *
import numpy as np
import pandas as pd
import plotly.express as px
import math

frame = Frame(path="/Users/lvmeizhong/Downloads/expoly-with-legacy/Alpoly_elongate.dream3d")
gid = 883
# df = fr.from_ID_to_D(gid)
# print(np.shape(fr.GrainId))
# print(fr.HZ_lim)
out_margin = frame.find_grain_NN_with_out(gid)
extend = frame.get_extend_Out_(out_margin, 3)
extend = frame.renew_outer_margin(extend)


# idx = np.argwhere(fr.GrainId == gid)
# df = pd.DataFrame(idx[:, :3], columns=['HZ', 'HY', 'HX'])
print(extend)
extend_xyz = extend.rename(columns={"grain-ID": "ID"}).copy()

cfg = CarveConfig(
    lattice="DIA",          # "FCC" / "BCC" / "DIA"
    ratio=3,              # 晶格间距（你原来常用 1.5）
    ci_radius=math.sqrt(2), # 与你老脚本一致的邻域半径
    random_center=False,    # 是否用随机球心；调试建议先关掉
    rng_seed=42,            # 可复现（只有在 random_center=True 时有意义）
)



extended = process_extend(883, frame, cfg)

# print("shape:", df.shape, "columns:", list(df.columns))
# print(df.head(20).to_string(index=False))
# print("min(HX,HY,HZ):", df[["HX","HY","HZ"]].min().to_dict(),
#       "max(HX,HY,HZ):", df[["HX","HY","HZ"]].max().to_dict())

fig_draw = px.scatter_3d(extended, x='X', y='Y', z='Z',opacity=0.5)
fig_draw.show()