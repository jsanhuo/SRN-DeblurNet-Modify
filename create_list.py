import os

def loadTrainList():
    fp = open("train_list.txt", 'w');
    path = os.getcwd()
    path += "\\GOPRO_Large\\train"
    print(path)
    print(os.listdir(path))
    count = 0
    for f in os.listdir(path):
        cpath = path+"\\"+f+"\\sharp"
        bpath = path+"\\"+f+"\\blur"
        for k in os.listdir(cpath):
            s = cpath+"\\"+k+" "+bpath+"\\"+k+"\n"
            fp.write(s)
            count+=1
    print(count)


def loadTestList():
    fp = open("test_list.txt", 'w');
    path = os.getcwd()
    path += "\\GOPRO_Large\\test"
    print(path)
    print(os.listdir(path))
    count = 0
    for f in os.listdir(path):
        cpath = path+"\\"+f+"\\sharp"
        bpath = path+"\\"+f+"\\blur"
        for k in os.listdir(cpath):
            s = cpath+"\\"+k+" "+bpath+"\\"+k+"\n"
            fp.write(s)
            count+=1
    print(count)

if __name__ == '__main__':
    loadTestList()
