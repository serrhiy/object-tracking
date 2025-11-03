import os

RESOURCES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources")
)


def main():
    print(RESOURCES_DIR)


if __name__ == "__main__":
    main()
