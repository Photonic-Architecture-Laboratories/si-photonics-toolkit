def readme():
    """
    Read README.md
    :return:
    """
    with open(".\\README.md") as file:
        return file.read()


def main():
    """
    Print out content in README.md
    :return:
    """
    print(readme())
