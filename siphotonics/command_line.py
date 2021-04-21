def readme():
    with open(".\\README.md") as f:
        return f.read()


def main():
    print(readme())
