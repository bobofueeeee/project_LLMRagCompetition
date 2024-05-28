import subprocess


def export_pip_packages_to_txt(output_file):
    # 使用pip list命令获取已安装包列表
    result = subprocess.run(['pip', 'list', '--format=freeze'], stdout=subprocess.PIPE, text=True)

    # 检查命令是否成功执行
    if result.returncode != 0:
        print(f"Error executing pip list: {result.stderr}")
        return

        # 将输出写入到文件
    with open(output_file, 'w') as f:
        f.write(result.stdout)

    # 调用函数并指定输出文件的名称


export_pip_packages_to_txt('pip_packages.txt')