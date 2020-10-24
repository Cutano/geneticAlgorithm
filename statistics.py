import numpy as np
import os
from PIL import Image
import plotly.graph_objects as go


def get_diff(mat_tar, mat_curr):
    return np.linalg.norm(abs(mat_tar - mat_curr))  # 以向量距离衡量差距


if __name__ == '__main__':
    pics_path = 'result/'
    target_pic_path = 'edge_small.png'
    fileList = os.listdir(pics_path)
    gens = [int(x[x.find('n') + 1:x.find('.')]) for x in fileList]
    gens.sort()
    fileList.sort(key=lambda x: int(x[x.find('n') + 1:x.find('.')]))
    pics = [np.asarray(Image.open(pics_path + file), dtype=np.int32) for file in fileList]
    tar_img = np.asarray(Image.open(target_pic_path), dtype=np.int32)
    diffs = [get_diff(tar_img, pic) for pic in pics]
    dic = (zip(gens, diffs))
    for pic in pics:
        print(get_diff(tar_img, pic))
    trace = go.Scatter(
        x=gens,
        y=diffs,
        mode='markers',
        marker=dict(
            size=16,
            color=np.random.randn(500),
            colorscale='Viridis',
            showscale=True
        )
    )
    trace2 = go.Scatter(
        x=gens,
        y=diffs,
        mode='lines',
        name='lines'
    )
    fig = go.Figure(trace)
    fig.show()