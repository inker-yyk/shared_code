import os
import pandas as pd

def convert_txt_to_excel(input_path, output_path):
    """
    将指定格式的TXT文件转换为Excel文件，并计算每个指标在所有类别中的平均值。

    参数:
    input_path (str): 输入的TXT文件路径。
    output_path (str): 输出的Excel文件路径。
    """
    # 读取文件内容
    with open(input_path, 'r') as file:
        lines = file.readlines()

    # 初始化数据存储
    data = []
    current_category = None

    # 处理每一行内容
    for line in lines:
        line = line.strip()
        if line.startswith('--------------------------'):
            continue
        elif ',' in line:
            metric, network, value = line.split(',')
            value = float(value)
            data.append([current_category, metric, network, value])
        else:
            current_category = line

    # 将数据转换为DataFrame
    df = pd.DataFrame(data, columns=['Category', 'Metric', 'Network', 'Value'])

    # 计算每个指标在所有类别中的平均值
    overall_avg_df = df.groupby('Metric')['Value'].mean().reset_index()
    overall_avg_df['Category'] = 'Overall Average'
    overall_avg_df['Network'] = 'average'

    # 将原始数据和平均值合并
    final_df = pd.concat([df, overall_avg_df], ignore_index=True)

    # 使用openpyxl进行写入和格式化
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        final_df.to_excel(writer, index=False, sheet_name='Data')

        # 获取工作表
        worksheet = writer.sheets['Data']

        # 自动调整列宽
        for col in worksheet.columns:
            max_length = 0
            column = col[0].column_letter  # 获取列字母
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column].width = adjusted_width

    print(f"Data has been successfully saved to {output_path} with overall averages calculated.")


if __name__ == '__main__':
    source_path = "/hdd2/yyk/DiffAD-main_4/logs/img_feature_817/"
    file_name  = ["ans_v0.txt", "ans_v1.txt", "ans_v2.txt", "ans_v3.txt"]
    output_name = "ans_excel_{}.xlsx"
    for i,name in enumerate(file_name):
        input_path = os.path.join(source_path, name)
        output_path = os.path.join(source_path, output_name.format(str(i)))
        print(input_path)
        convert_txt_to_excel(input_path, output_path)
