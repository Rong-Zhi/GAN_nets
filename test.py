import constant as c
import generator_model as classG

def main():
    # c.get_dir("Helloword")
    # print(c.TEST_DIR)
    # print(c.DATA_DIR)
    # print(c.TRAIN_DIR)
    # print(c.get_test_dims(c.TEST_DIR))
    generator = classG.Generator()
    generator.generator()



if __name__ == "__main__":
    main()