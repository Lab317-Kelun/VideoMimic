#!/usr/bin/env python3
"""
H5动作数据验证脚本
验证H5文件是否符合VideoMimic训练标准
"""

import h5py
import numpy as np
import sys
import os
from pathlib import Path

def validate_h5_file(file_path):
    """验证单个H5文件"""
    print(f"\n{'='*80}")
    print(f"验证文件: {file_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件不存在 - {file_path}")
        return False
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"✅ 文件打开成功")
            
            # 1. 检查必需字段
            required_fields = ['root_pos', 'root_quat', 'joints']
            print(f"\n📋 检查必需字段:")
            for field in required_fields:
                if field in f:
                    data = f[field]
                    print(f"  ✅ {field}: shape={data.shape}, dtype={data.dtype}")
                else:
                    print(f"  ❌ {field}: 缺失")
                    return False
            
            # 2. 检查可选字段
            optional_fields = ['link_pos', 'link_quat', 'contacts']
            print(f"\n📋 检查可选字段:")
            for field in optional_fields:
                if field in f:
                    data = f[field]
                    if hasattr(data, 'shape'):
                        print(f"  ✅ {field}: shape={data.shape}, dtype={data.dtype}")
                    else:
                        print(f"  ✅ {field}: {type(data).__name__} (组)")
                else:
                    print(f"  ⚠️  {field}: 缺失 (可选)")
            
            # 3. 检查属性
            print(f"\n📋 检查属性:")
            attrs_to_check = ['/joint_names', '/link_names', '/fps']
            for attr in attrs_to_check:
                if attr in f.attrs:
                    value = f.attrs[attr]
                    print(f"  ✅ {attr}: {value}")
                else:
                    print(f"  ⚠️  {attr}: 缺失 (可选)")
            
            # 4. 详细数据验证
            print(f"\n🔍 详细数据验证:")
            
            # 验证root_pos
            root_pos = f['root_pos'][:]
            print(f"  root_pos: shape={root_pos.shape}")
            print(f"    - 数值范围: X[{root_pos[:, 0].min():.3f}, {root_pos[:, 0].max():.3f}], "
                  f"Y[{root_pos[:, 1].min():.3f}, {root_pos[:, 1].max():.3f}], "
                  f"Z[{root_pos[:, 2].min():.3f}, {root_pos[:, 2].max():.3f}]")
            
            # 验证root_quat
            root_quat = f['root_quat'][:]
            print(f"  root_quat: shape={root_quat.shape}")
            quat_norms = np.linalg.norm(root_quat, axis=1)
            print(f"    - 四元数模长范围: [{quat_norms.min():.6f}, {quat_norms.max():.6f}]")
            if np.allclose(quat_norms, 1.0, atol=1e-3):
                print(f"    ✅ 四元数已归一化")
            else:
                print(f"    ❌ 四元数未归一化")
                return False
            
            # 验证joints
            joints = f['joints'][:]
            print(f"  joints: shape={joints.shape}")
            print(f"    - 关节角度范围: [{joints.min():.3f}, {joints.max():.3f}] (弧度)")
            if joints.shape[1] == 23:
                print(f"    ✅ 关节数量正确 (23个)")
            else:
                print(f"    ❌ 关节数量错误: 期望23个, 实际{joints.shape[1]}个")
                return False
            
            # 验证link_pos (如果存在)
            if 'link_pos' in f:
                link_pos = f['link_pos'][:]
                print(f"  link_pos: shape={link_pos.shape}")
                if link_pos.shape[2] == 3:
                    print(f"    ✅ 位置维度正确 (3维)")
                else:
                    print(f"    ❌ 位置维度错误: 期望3维, 实际{link_pos.shape[2]}维")
                    return False
                
                if link_pos.shape[1] == 13:
                    print(f"    ✅ 身体部位数量正确 (13个)")
                else:
                    print(f"    ⚠️  身体部位数量: 期望13个, 实际{link_pos.shape[1]}个")
            
            # 验证link_quat (如果存在)
            if 'link_quat' in f:
                link_quat = f['link_quat'][:]
                print(f"  link_quat: shape={link_quat.shape}")
                quat_norms = np.linalg.norm(link_quat, axis=2)
                print(f"    - 四元数模长范围: [{quat_norms.min():.6f}, {quat_norms.max():.6f}]")
                if np.allclose(quat_norms, 1.0, atol=1e-3):
                    print(f"    ✅ 四元数已归一化")
                else:
                    print(f"    ❌ 四元数未归一化")
                    return False
            
            # 验证contacts (如果存在)
            if 'contacts' in f:
                contacts_group = f['contacts']
                print(f"  contacts: 包含 {len(contacts_group)} 个接触点")
                for contact_name in contacts_group:
                    contact_data = contacts_group[contact_name][:]
                    print(f"    - {contact_name}: shape={contact_data.shape}, dtype={contact_data.dtype}")
            
            # 5. 检查数据一致性
            print(f"\n�� 数据一致性检查:")
            num_frames = root_pos.shape[0]
            print(f"  总帧数: {num_frames}")
            
            if root_quat.shape[0] != num_frames:
                print(f"  ❌ root_quat帧数不匹配: {root_quat.shape[0]} != {num_frames}")
                return False
            else:
                print(f"  ✅ root_quat帧数匹配")
            
            if joints.shape[0] != num_frames:
                print(f"  ❌ joints帧数不匹配: {joints.shape[0]} != {num_frames}")
                return False
            else:
                print(f"  ✅ joints帧数匹配")
            
            if 'link_pos' in f and f['link_pos'].shape[0] != num_frames:
                print(f"  ❌ link_pos帧数不匹配: {f['link_pos'].shape[0]} != {num_frames}")
                return False
            elif 'link_pos' in f:
                print(f"  ✅ link_pos帧数匹配")
            
            # 6. 计算帧率
            if '/fps' in f.attrs:
                fps = f.attrs['/fps']
                duration = num_frames / fps
                print(f"\n⏱️  时间信息:")
                print(f"  帧率: {fps} fps")
                print(f"  总时长: {duration:.2f} 秒")
                print(f"  总帧数: {num_frames}")
            else:
                print(f"\n⏱️  时间信息:")
                print(f"  ⚠️  未找到帧率信息")
                print(f"  总帧数: {num_frames}")
            
            print(f"\n✅ 文件验证通过!")
            return True
            
    except Exception as e:
        print(f"❌ 错误: 无法读取文件 - {e}")
        return False

def main():
    """主函数"""
    print("VideoMimic H5数据验证工具")
    print("=" * 80)
    
    # 要验证的文件列表
    files_to_validate = [
        "/home/asus/VideoMimic/simulation/data/videomimic_captures/videomimic_captures/megahunter_align3r_reconstruction_results_apr17_IMG_0945_cam01_frame_0_300_subsample_2.pkl/retarget_poses_g1.h5",
        "/home/asus/VideoMimic/simulation/data/videomimic_captures/videomimic_captures/A_test/retarget_poses_g1.h5",
        "/home/asus/VideoMimic/simulation/data/videomimic_captures/videomimic_captures/megahunter_align3r_reconstruction_results_apr21_IMG_6327_cam01_frame_0_300_subsample_2.pkl/retarget_poses_g1.h5",
        "/home/asus/VideoMimic/simulation/data/videomimic_captures/videomimic_captures/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl/retarget_poses_g1.h5"
    ]
    
    # 检查文件是否存在
    existing_files = []
    for file_path in files_to_validate:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    if not existing_files:
        print("❌ 没有找到任何要验证的文件")
        return
    
    # 验证每个文件
    all_valid = True
    for file_path in existing_files:
        is_valid = validate_h5_file(file_path)
        if not is_valid:
            all_valid = False
    
    # 总结
    print(f"\n{'='*80}")
    print("验证总结:")
    if all_valid:
        print("✅ 所有文件验证通过,符合训练标准!")
    else:
        print("❌ 部分文件验证失败,请检查数据格式!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

'''
1. 'root_pos': numpy.ndarray, shape=(N, 3), dtype=float32 或 float64
   - N: 动作帧数(例如300帧)
   - 3: [X, Y, Z] 世界坐标系下的位置(单位:米)
   - 示例:shape=(300, 3),表示300帧动作,每帧有XYZ三个坐标
   - 数值范围:通常 X,Y ∈ [-10, 10], Z ∈ [0, 2](地面以上)

2. 'root_quat': numpy.ndarray, shape=(N, 4), dtype=float32 或 float64
   - N: 动作帧数(与root_pos一致)
   - 4: 四元数 [W, X, Y, Z](WXYZ格式,注意不是XYZW)
   - 示例:shape=(300, 4)
   - 数值要求:必须是单位四元数(模长=1),即 W²+X²+Y²+Z²=1
   - 坐标系:Z轴向上,右手坐标系

3. 'joints': numpy.ndarray, shape=(N, 23), dtype=float32 或 float64
   - N: 动作帧数(与root_pos一致)
   - 23: G1机器人的23个关节角度(单位:弧度)
   - 示例:shape=(300, 23)
   - 数值范围:每个关节 ∈ [-π, π](-3.14159 到 3.14159)
   - 关节顺序(必须严格按照此顺序):
     索引 0-5:   左腿(left_hip_yaw, left_hip_roll, left_hip_pitch, 
                       left_knee, left_ankle_pitch, left_ankle_roll)
     索引 6-11:  右腿(right_hip_yaw, right_hip_roll, right_hip_pitch,
                       right_knee, right_ankle_pitch, right_ankle_roll)
     索引 12:    躯干(torso)
     索引 13-15: 腰部(waist_yaw, waist_pitch, waist_roll)
     索引 16-19: 左臂(left_shoulder_pitch, left_shoulder_roll, 
                       left_shoulder_yaw, left_elbow)
     索引 20-22: 右臂(right_shoulder_pitch, right_shoulder_roll,
                       right_shoulder_yaw, right_elbow)

【可选但强烈推荐的字段】

4. 'link_pos': numpy.ndarray, shape=(N, num_links, 3), dtype=float32 或 float64
   - N: 动作帧数
   - num_links: 跟踪的身体部位数量(通常13个)
   - 3: [X, Y, Z] 世界坐标下的位置(单位:米)
   - 示例:shape=(300, 13, 3)
   - 如果不提供,训练时link_pos_tracking奖励会失效(但不会报错)
   - 13个跟踪部位(按顺序):
     0: pelvis(骨盆)
     1: left_hip_pitch_link(左髋俯仰)
     2: left_knee_link(左膝)
     3: left_ankle_roll_link(左踝滚转)
     4: right_hip_pitch_link(右髋俯仰)
     5: right_knee_link(右膝)
     6: right_ankle_roll_link(右踝滚转)
     7: left_shoulder_pitch_link(左肩俯仰)
     8: left_elbow_link(左肘)
     9: left_wrist_yaw_link(左腕偏航)
     10: right_shoulder_pitch_link(右肩俯仰)
     11: right_elbow_link(右肘)
     12: right_wrist_yaw_link(右腕偏航)

5. 'link_quat': numpy.ndarray, shape=(N, num_links, 4), dtype=float32 或 float64
   - N: 动作帧数
   - num_links: 身体部位数量(与link_pos一致,通常13个)
   - 4: 四元数 [W, X, Y, Z](WXYZ格式)
   - 示例:shape=(300, 13, 4)
   - 每个四元数必须是单位四元数(模长=1)
   - 如果不提供,link方向相关的奖励会失效

6. 'contacts': 字典(dict),包含接触信息
   - 'left_foot': numpy.ndarray, shape=(N,), dtype=bool 或 int 或 float
     * N: 动作帧数
     * 值: 0=不接触地面,1=接触地面
     * 示例:shape=(300,),表示300帧中每帧左脚是否接触
   - 'right_foot': numpy.ndarray, shape=(N,), dtype=bool 或 int 或 float
     * N: 动作帧数
     * 值: 0=不接触,1=接触
     * 示例:shape=(300,)
   - 如果不提供,feet_contact_matching奖励会失效(但不影响训练)

【H5属性(attributes)】

7. '/joint_names': 字符串列表或numpy数组(存储为H5属性)
   - 类型:list[str] 或 numpy.ndarray of strings
   - 长度:23(与joints的第二维度一致)
   - 示例:['left_hip_yaw_joint', 'left_hip_roll_joint', ...]
   - 必须按照上面joints中定义的顺序
   - 在H5中存储方式:data.attrs['/joint_names'] = joint_names_list

8. '/link_names': 字符串列表(存储为H5属性)
   - 类型:list[str] 或 numpy.ndarray of strings
   - 长度:13(如果有link_pos/link_quat的话)
   - 示例:['pelvis', 'left_hip_pitch_link', 'left_knee_link', ...]
   - 必须按照上面link_pos中定义的顺序
   - 在H5中存储方式:data.attrs['/link_names'] = link_names_list

9. '/fps': 浮点数(存储为H5属性)
   - 类型:float 或 int
   - 示例值:30.0, 60.0
   - 表示动作数据的帧率(帧/秒)
   - 在H5中存储方式:data.attrs['/fps'] = 30.0
   - 如果不提供,会使用default_data_fps_override或default_data_fps
'''