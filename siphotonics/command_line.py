def readme():
    with open(".\\README.rst") as f:
        return f.read()

def main():
    print(readme())
