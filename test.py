from pkg_resources import ensure_directory
from utils import *
import argparse
from model import *
import tqdm
import numpy as np
import pandas as pd
import configparser
import ast

# import matplotlib.pyplot as plt
# import seaborn as sns

DATASET = "BJTaxi"  # PEMSD4 or PEMSD8

config_file = "./{}.conf".format(DATASET)
config = configparser.ConfigParser()
config.read(config_file)

parser = argparse.ArgumentParser(description="arguments")
parser.add_argument("--no_cuda", action="store_true", help="没有GPU")
parser.add_argument(
    "--data", type=str, default=config["data"]["data"], help="data path"
)
parser.add_argument(
    "--sensors_distance",
    type=str,
    default=config["data"]["sensors_distance"],
    help="节点距离文件",
)
parser.add_argument(
    "--column_wise",
    type=eval,
    default=config["data"]["column_wise"],
    help="是指列元素的级别上进行归一，否则是全样本取值",
)
parser.add_argument(
    "--normalizer", type=str, default=config["data"]["normalizer"], help="归一化方式"
)
parser.add_argument(
    "--batch_size", type=int, default=config["data"]["batch_size"], help="batch大小"
)

parser.add_argument(
    "--num_of_vertices",
    type=int,
    default=config["model"]["num_of_vertices"],
    help="传感器数量",
)
parser.add_argument(
    "--construct_type",
    type=str,
    default=config["model"]["construct_type"],
    help="构图方式  {connectivity, distance}",
)
parser.add_argument(
    "--in_dim", type=int, default=config["model"]["in_dim"], help="输入维度"
)
parser.add_argument(
    "--hidden_dims",
    type=list,
    default=ast.literal_eval(config["model"]["hidden_dims"]),
    help="中间各STSGCL层的卷积操作维度",
)
parser.add_argument(
    "--first_layer_embedding_size",
    type=int,
    default=config["model"]["first_layer_embedding_size"],
    help="第一层输入层的维度",
)
parser.add_argument(
    "--out_layer_dim",
    type=int,
    default=config["model"]["out_layer_dim"],
    help="输出模块中间层维度",
)
parser.add_argument(
    "--history", type=int, default=config["model"]["history"], help="每个样本输入的离散时序"
)
parser.add_argument(
    "--horizon", type=int, default=config["model"]["horizon"], help="每个样本输出的离散时序"
)
parser.add_argument(
    "--strides",
    type=int,
    default=config["model"]["strides"],
    help="滑动窗口步长，local时空图使用几个时间步构建的，默认为3",
)
parser.add_argument(
    "--temporal_emb",
    type=eval,
    default=config["model"]["temporal_emb"],
    help="是否使用时间嵌入向量",
)
parser.add_argument(
    "--spatial_emb",
    type=eval,
    default=config["model"]["spatial_emb"],
    help="是否使用空间嵌入向量",
)
parser.add_argument(
    "--use_mask", type=eval, default=config["model"]["use_mask"], help="是否使用mask矩阵优化adj"
)
parser.add_argument(
    "--activation",
    type=str,
    default=config["model"]["activation"],
    help="激活函数 {relu, GlU}",
)

parser.add_argument("--log_file", default=config["test"]["log_file"], help="log file")
parser.add_argument("--checkpoint", type=str, help="")

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


log = open(args.log_file, "w")
log_string(log, str(args))


def main():
    adj = np.load(args.sensors_distance)["adj_mx"]

    local_adj = construct_adj(A=adj, steps=args.strides)

    local_adj = torch.FloatTensor(local_adj)

    dataloader = load_dataset(
        dataset_dir=args.data,
        normalizer=args.normalizer,
        batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        column_wise=args.column_wise,
    )

    scaler = dataloader["scaler"]

    model = STSGCN(
        adj=local_adj,
        history=args.history,
        num_of_vertices=args.num_of_vertices,
        in_dim=args.in_dim,
        hidden_dims=args.hidden_dims,
        first_layer_embedding_size=args.first_layer_embedding_size,
        out_layer_dim=args.out_layer_dim,
        activation=args.activation,
        use_mask=args.use_mask,
        temporal_emb=args.temporal_emb,
        spatial_emb=args.spatial_emb,
        horizon=args.horizon,
        strides=args.strides,
    ).to(device)

    # model.load_state_dict(torch.load(args.checkpoint))  # 多GPU保存，多GPU加载(但可能会导致后面没法画图)，但GPU保存，单GPU加载

    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in torch.load(args.checkpoint).items()}
    )
    # 多GPU保存， 单GPU加载

    model.eval()

    log_string(log, "加载模型成功")

    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)

    # realy = realy[..., 0]
    # [B, T, N]

    for iter, (x, y) in tqdm.tqdm(enumerate(dataloader["test_loader"].get_iterator())):
        testx = torch.Tensor(x).to(device)
        with torch.no_grad():
            preds = model(testx)
            # [B, T, N]

            outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[
        : realy.size(0), ...
    ]  # 这里是因为你在做batch的时候，可能会padding出新的sample以满足batch_size的要求

    pred = scaler.inverse_transform(yhat)
    real = realy

    log_string(log, f"mae: {masked_mae_torch(pred, real, mask_value=0)}")
    log_string(log, f"mape: {masked_mape_torch(pred, real, mask_value=0)}")
    log_string(log, f"mse: {masked_mse_torch(pred, real, mask_value=0)}")
    log_string(log, f"rmse: {masked_rmse_torch(pred, real, mask_value=0)}")
    log_string(log, f"masked_mae: {masked_mae_torch(pred, real, 0.0, mask_value=5)}")
    log_string(log, f"masked_mape: {masked_mape_torch(pred, real, 0.0, mask_value=5)}")
    log_string(log, f"masked_mse: {masked_mse_torch(pred, real, 0.0, mask_value=5)}")
    log_string(log, f"masked_rmse: {masked_rmse_torch(pred, real, 0.0, mask_value=5)}")

    result = {
        "mae": masked_mae_torch(pred, real, mask_value=0).item(),
        "mape": masked_mape_torch(pred, real, mask_value=0).item(),
        "mse": masked_mse_torch(pred, real, mask_value=0).item(),
        "rmse": masked_rmse_torch(pred, real, mask_value=0).item(),
        "masked_mae": masked_mae_torch(pred, real, 0.0, mask_value=5).item(),
        "masked_mape": masked_mape_torch(pred, real, 0.0, mask_value=5).item(),
        "masked_mse": masked_mse_torch(pred, real, 0.0, mask_value=5).item(),
        "masked_rmse": masked_rmse_torch(pred, real, 0.0, mask_value=5).item(),
    }

    save_folder = config["test"]["log_folder"]
    save_csv_path = os.path.join(save_folder, f"metric.csv")
    save_npz_path = os.path.join(save_folder, f"prediction.npz")
    ensure_directory(save_csv_path)
    # convert the dictionary to a pandas dataframe
    df = pd.DataFrame(list(result.items()), columns=["key", "value"])

    # save the dataframe as a CSV file with the column name as "key"
    df.to_csv(os.path.join(save_csv_path), index=False)

    np.savez(save_npz_path, prediction=pred.cpu().numpy(), truth=real.cpu().numpy())


if __name__ == "__main__":
    main()
    log.close()
