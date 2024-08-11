import subprocess


def test_train_test():
    process = subprocess.run(
        ["python", "train.py", "config/dummy.yaml", "use_cpu=true"],
        stdout=subprocess.PIPE,
    )
    assert process.returncode == 0
    output = process.stdout.decode("utf-8")

    output_cmd = None
    output_rows = output.split("\n")
    for line in output_rows:
        if "test cmd:" in line:
            output_cmd = line
            break
    assert output_cmd is not None

    process = subprocess.run(
        output_cmd.split(" "),
        stdout=subprocess.PIPE,
    )
    assert process.returncode == 0
