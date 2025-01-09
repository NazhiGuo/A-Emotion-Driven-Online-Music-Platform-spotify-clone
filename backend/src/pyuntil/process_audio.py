import opensmile
import librosa
import pandas as pd
import soundfile as sf
import sys
import json
import joblib
import os
import tensorflow as tf
import numpy as np
common_df = ['frameTime',
    'F0final_sma_stddev', 'F0final_sma_amean', 'voicingFinalUnclipped_sma_stddev',
    'voicingFinalUnclipped_sma_amean', 'jitterLocal_sma_stddev', 'jitterLocal_sma_amean',
    'jitterDDP_sma_stddev', 'jitterDDP_sma_amean', 'shimmerLocal_sma_stddev',
    'shimmerLocal_sma_amean', 'logHNR_sma_stddev', 'logHNR_sma_amean',
    'audspec_lengthL1norm_sma_stddev', 'audspec_lengthL1norm_sma_amean',
    'audspecRasta_lengthL1norm_sma_stddev', 'audspecRasta_lengthL1norm_sma_amean',
    'pcm_RMSenergy_sma_stddev', 'pcm_RMSenergy_sma_amean', 'pcm_zcr_sma_stddev',
    'pcm_zcr_sma_amean', 'audSpec_Rfilt_sma[0]_stddev', 'audSpec_Rfilt_sma[0]_amean',
    'audSpec_Rfilt_sma[1]_stddev', 'audSpec_Rfilt_sma[1]_amean', 'audSpec_Rfilt_sma[2]_stddev',
    'audSpec_Rfilt_sma[2]_amean', 'audSpec_Rfilt_sma[3]_stddev', 'audSpec_Rfilt_sma[3]_amean',
    'audSpec_Rfilt_sma[4]_stddev', 'audSpec_Rfilt_sma[4]_amean', 'audSpec_Rfilt_sma[5]_stddev',
    'audSpec_Rfilt_sma[5]_amean', 'audSpec_Rfilt_sma[6]_stddev', 'audSpec_Rfilt_sma[6]_amean',
    'audSpec_Rfilt_sma[7]_stddev', 'audSpec_Rfilt_sma[7]_amean', 'audSpec_Rfilt_sma[8]_stddev',
    'audSpec_Rfilt_sma[8]_amean', 'audSpec_Rfilt_sma[9]_stddev', 'audSpec_Rfilt_sma[9]_amean',
    'audSpec_Rfilt_sma[10]_stddev', 'audSpec_Rfilt_sma[10]_amean', 'audSpec_Rfilt_sma[11]_stddev',
    'audSpec_Rfilt_sma[11]_amean', 'audSpec_Rfilt_sma[12]_stddev', 'audSpec_Rfilt_sma[12]_amean',
    'audSpec_Rfilt_sma[13]_stddev', 'audSpec_Rfilt_sma[13]_amean', 'audSpec_Rfilt_sma[14]_stddev',
    'audSpec_Rfilt_sma[14]_amean', 'audSpec_Rfilt_sma[15]_stddev', 'audSpec_Rfilt_sma[15]_amean',
    'audSpec_Rfilt_sma[16]_stddev', 'audSpec_Rfilt_sma[16]_amean', 'audSpec_Rfilt_sma[17]_stddev',
    'audSpec_Rfilt_sma[17]_amean', 'audSpec_Rfilt_sma[18]_stddev', 'audSpec_Rfilt_sma[18]_amean',
    'audSpec_Rfilt_sma[19]_stddev', 'audSpec_Rfilt_sma[19]_amean', 'audSpec_Rfilt_sma[20]_stddev',
    'audSpec_Rfilt_sma[20]_amean', 'audSpec_Rfilt_sma[21]_stddev', 'audSpec_Rfilt_sma[21]_amean',
    'audSpec_Rfilt_sma[22]_stddev', 'audSpec_Rfilt_sma[22]_amean', 'audSpec_Rfilt_sma[23]_stddev',
    'audSpec_Rfilt_sma[23]_amean', 'audSpec_Rfilt_sma[24]_stddev', 'audSpec_Rfilt_sma[24]_amean',
    'audSpec_Rfilt_sma[25]_stddev', 'audSpec_Rfilt_sma[25]_amean', 'pcm_fftMag_fband250-650_sma_stddev',
    'pcm_fftMag_fband250-650_sma_amean', 'pcm_fftMag_fband1000-4000_sma_stddev',
    'pcm_fftMag_fband1000-4000_sma_amean', 'pcm_fftMag_spectralRollOff25.0_sma_stddev',
    'pcm_fftMag_spectralRollOff25.0_sma_amean', 'pcm_fftMag_spectralRollOff50.0_sma_stddev',
    'pcm_fftMag_spectralRollOff50.0_sma_amean', 'pcm_fftMag_spectralRollOff75.0_sma_stddev',
    'pcm_fftMag_spectralRollOff75.0_sma_amean', 'pcm_fftMag_spectralRollOff90.0_sma_stddev',
    'pcm_fftMag_spectralRollOff90.0_sma_amean', 'pcm_fftMag_spectralFlux_sma_stddev',
    'pcm_fftMag_spectralFlux_sma_amean', 'pcm_fftMag_spectralCentroid_sma_stddev',
    'pcm_fftMag_spectralCentroid_sma_amean', 'pcm_fftMag_spectralEntropy_sma_stddev',
    'pcm_fftMag_spectralEntropy_sma_amean', 'pcm_fftMag_spectralVariance_sma_stddev',
    'pcm_fftMag_spectralVariance_sma_amean', 'pcm_fftMag_spectralSkewness_sma_stddev',
    'pcm_fftMag_spectralSkewness_sma_amean', 'pcm_fftMag_spectralKurtosis_sma_stddev',
    'pcm_fftMag_spectralKurtosis_sma_amean', 'pcm_fftMag_spectralSlope_sma_stddev',
    'pcm_fftMag_spectralSlope_sma_amean', 'pcm_fftMag_psySharpness_sma_stddev',
    'pcm_fftMag_psySharpness_sma_amean', 'pcm_fftMag_spectralHarmonicity_sma_stddev',
    'pcm_fftMag_spectralHarmonicity_sma_amean', 'F0final_sma_de_stddev', 'F0final_sma_de_amean',
    'voicingFinalUnclipped_sma_de_stddev', 'voicingFinalUnclipped_sma_de_amean',
    'jitterLocal_sma_de_stddev', 'jitterLocal_sma_de_amean', 'jitterDDP_sma_de_stddev',
    'jitterDDP_sma_de_amean', 'shimmerLocal_sma_de_stddev', 'shimmerLocal_sma_de_amean',
    'logHNR_sma_de_stddev', 'logHNR_sma_de_amean', 'audspec_lengthL1norm_sma_de_stddev',
    'audspecRasta_lengthL1norm_sma_de_stddev', 'pcm_RMSenergy_sma_de_stddev',
    'pcm_zcr_sma_de_stddev', 'audSpec_Rfilt_sma_de[0]_stddev', 'audSpec_Rfilt_sma_de[1]_stddev',
    'audSpec_Rfilt_sma_de[2]_stddev', 'audSpec_Rfilt_sma_de[3]_stddev', 'audSpec_Rfilt_sma_de[4]_stddev',
    'audSpec_Rfilt_sma_de[5]_stddev', 'audSpec_Rfilt_sma_de[6]_stddev', 'audSpec_Rfilt_sma_de[7]_stddev',
    'audSpec_Rfilt_sma_de[8]_stddev', 'audSpec_Rfilt_sma_de[9]_stddev', 'audSpec_Rfilt_sma_de[10]_stddev',
    'audSpec_Rfilt_sma_de[11]_stddev', 'audSpec_Rfilt_sma_de[12]_stddev', 'audSpec_Rfilt_sma_de[13]_stddev',
    'audSpec_Rfilt_sma_de[14]_stddev', 'audSpec_Rfilt_sma_de[15]_stddev', 'audSpec_Rfilt_sma_de[16]_stddev',
    'audSpec_Rfilt_sma_de[17]_stddev', 'audSpec_Rfilt_sma_de[18]_stddev', 'audSpec_Rfilt_sma_de[19]_stddev',
    'audSpec_Rfilt_sma_de[20]_stddev', 'audSpec_Rfilt_sma_de[21]_stddev', 'audSpec_Rfilt_sma_de[22]_stddev',
    'audSpec_Rfilt_sma_de[23]_stddev', 'audSpec_Rfilt_sma_de[24]_stddev', 'audSpec_Rfilt_sma_de[25]_stddev',
    'pcm_fftMag_fband250-650_sma_de_stddev', 'pcm_fftMag_fband1000-4000_sma_de_stddev',
    'pcm_fftMag_spectralRollOff25.0_sma_de_stddev', 'pcm_fftMag_spectralRollOff50.0_sma_de_stddev',
    'pcm_fftMag_spectralRollOff75.0_sma_de_stddev', 'pcm_fftMag_spectralRollOff90.0_sma_de_stddev',
    'pcm_fftMag_spectralFlux_sma_de_stddev', 'pcm_fftMag_spectralCentroid_sma_de_stddev',
    'pcm_fftMag_spectralEntropy_sma_de_stddev', 'pcm_fftMag_spectralVariance_sma_de_stddev',
    'pcm_fftMag_spectralSkewness_sma_de_stddev', 'pcm_fftMag_spectralKurtosis_sma_de_stddev',
    'pcm_fftMag_spectralSlope_sma_de_stddev', 'pcm_fftMag_psySharpness_sma_de_stddev',
    'pcm_fftMag_spectralHarmonicity_sma_de_stddev'
]

def main(audio_file):
    try:
        # 初始化 opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals
        )

        # 定义需要的特征顺序（确保 common_df 是你预定义的特征顺序）
        desired_feature_order = common_df

        # 加载音频文件
        y, sr = librosa.load(audio_file, sr=None)

        # 设置每次提取的音频片段长度和间隔
        segment_length = 0.5  # 每段音频的时长为0.5秒
        segment_samples = int(segment_length * sr)  # 计算对应的采样点数

        # 切分音频并提取特征
        num_segments = len(y) // segment_samples  # 计算总共有多少个片段
        all_features = []  # 用于存储所有的特征

        for i in range(num_segments):
            # 获取每个片段的音频数据
            start_sample = i * segment_samples
            end_sample = start_sample + segment_samples
            segment = y[start_sample:end_sample]

            # 保存片段为临时文件
            temp_filename = f"temp_segment_{i}.wav"
            sf.write(temp_filename, segment, sr)

            # 提取该片段的特征
            features = smile.process_file(temp_filename)

            # 过滤只保留在 desired_feature_order 中的特征
            filtered_features = {k: features[k] for k in features.columns if k in desired_feature_order}

            # 清理每个特征，提取数值部分
            cleaned_features = {}
            for feature, value in filtered_features.items():
                # 提取数值部分
                cleaned_features[feature] = value.iloc[0] if isinstance(value, pd.Series) else value
            start_time = i * segment_length

            # 将起始时间添加到特征字典中
            cleaned_features['frameTime'] = start_time

            # 将清理后的特征添加到列表
            all_features.append(cleaned_features)

            # 删除临时文件
            os.remove(temp_filename)

        # 汇总所有特征到一个DataFrame
        all_features_df = pd.DataFrame(all_features)

        # 按照预定的顺序重新排列特征列
        all_features_df = all_features_df[desired_feature_order]

        # 加载模型
        model_path = 'F:\Dowloadn\Music-Emotion-Recognition-Algorithm-main\MER_model.h5'
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
        feature_columns = all_features_df.columns[1:]
        X_pre = all_features_df[feature_columns].values
        print(X_pre.shape)
        # 加载标准化器
        scaler_path = 'F:\Dowloadn\Music-Emotion-Recognition-Algorithm-main\scaler.joblib'
        scaler = joblib.load(scaler_path)

        # 准备特征

        # 标准化特征
        X_scaled = scaler.transform(X_pre)

        # 重塑特征以适应模型输入
        X_reshaped = X_scaled.reshape(-1, 159, 1)

        # 预测
        predictions = loaded_model.predict(X_reshaped)

        # 生成结果
        result = []
        frame_times = np.arange(0, len(predictions) * 0.5, 0.5)
        # 遍历数据并构建字典
        for i in range(len(predictions)):
            entry = f"{{ frameTime: {round(frame_times[i], 1)}, valence: {round(float(predictions[i][1]) * 1000, 7)}, arousal: {round(float(predictions[i][0]) * 1000, 7)} }}"
            result.append(entry)

        # 将结果转换为 JavaScript 对象数组的字符串
        js_object_array = "[\n  " + ",\n  ".join(result) + "\n]"

        output_filename = "output.js"
        output_path = os.path.join(os.path.dirname(__file__), output_filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 将结果保存为 JavaScript 文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("const data = " + js_object_array + ";")
            f.write("\n\nexport default data;")

    except Exception as e:
        print(f"Error processing audio file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "需要一个音频文件路径作为参数"}))
        sys.exit(1)
    main(sys.argv[1])
