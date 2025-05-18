from modelscope.hub.api import HubApi
YOUR_ACCESS_TOKEN = 'd45f132a-252d-4702-a4ce-94780f0913cf'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

from modelscope.hub.constants import Licenses, ModelVisibility

################### 模型
# owner_name = 'hmdeng'
# model_name = 'mm2025'
# model_id = f"{owner_name}/{model_name}"

# api.create_model(
#     model_id,
#     visibility=ModelVisibility.PUBLIC,
#     license=Licenses.APACHE_V2,
#     chinese_name="我的测试模型"
# )

# 上传
# api.upload_folder(
#     repo_id=f"{owner_name}/{model_name}",
#     folder_path='work_dirs/lup1m_path-l_to_vit_tiny_from_cls_patch_atten_moe_v2/checkpoint0100.pth', # 本地的模型文件
#     path_in_repo='lup1m_path-l_to_vit_tiny_from_cls_patch_atten_moe_v2', # 传到modelscope上的文件夹名称，没有就不传这个参数
#     commit_message='lup1m epoch 100 path large cls patch atten to vit tiny', # 提交的信息
#     repo_type='model' # 如果是数据集就是'dataset'
# )
# 下载
# from modelscope import snapshot_download
# model_path =snapshot_download(
#     repo_id=f"{owner_name}/{model_name}",
#     repo_type='model',
#     local_dir='./work_dirs',
#     allow_patterns=['lup1m_solider-b_to_swin_tiny_from_cls_patch_moe/checkpoint.pth']
#     )

# ################### 数据集
owner_name = 'hmdeng'
dataset_name = 'LUP'
dataset_id = f"{owner_name}/{dataset_name}"

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='/mnt/hdd1/wangxuanhan/datasets/CrowdHuman.tar.gz',
    commit_message='coco',
    repo_type = 'dataset'
)