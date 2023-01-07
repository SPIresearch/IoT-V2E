import os
import shutil
#video path

# trainvideopath="/workspace/lustre/LD-Data/fiveclass_rs/"
trainvideopath = r"LD-Data/fc_rs_image"
trainlist=os.listdir(trainvideopath)
dict={"W":"1","N1":"2","N2":"3","N3":"4","R":"5"}

print("TrainList Start")

for idx_train in trainlist:
    subpath=trainvideopath + '/' + idx_train
    sublist=os.listdir(subpath)
    l=len(sublist)
    z=int(l*1)
    print(idx_train + ':' + str(z))
    for i in sublist:
       source=idx_train+"/"+i
       # print(source+" "+dict[idx_train] + "\n")
       with open(r"guorongxiao/EEG_Video_Fusion/MakeVideoList/trainlist01.txt" , 'a') as ff:
            ff.write( source+" "+ dict[idx_train] + "\n")
    # for a in sublist[z:]:
    #    source=idx+"/"+a
    #    with open("/workspace/lustre/LD-Data/annotation_dir_path/711_train/1testlist.txt" , 'a') as f:
    #         f.write( source+" "+dict[idx] + "\n")

testvideopath = r"LD-Data/20val_img_fiveclass_rs"
testlist=os.listdir(testvideopath)
dict={"W":"1","N1":"2","N2":"3","N3":"4","R":"5"}


print("TestList Start")

for idx_test in testlist:
    subpath=testvideopath + '/' + idx_test
    sublist=os.listdir(subpath)
    l=len(sublist)
    z=int(l*1)
    print(idx_test + ':' + str(z))
    for i in sublist:
       source=idx_test+"/"+i
       # print(source+" "+dict[idx_test] + "\n")
       with open(r"guorongxiao/EEG_Video_Fusion/MakeVideoList/testlist01.txt" , 'a') as pp:
            pp.write( source+" "+ dict[idx_test] + "\n")



# import os
# import shutil
# import numpy as np
# trainvideopath="/home/spi/20valvideo-rs/"
# list=os.listdir(trainvideopath)
# dict={"W":"1","N1":"2","N2":"3","N3":"4","R":"5"}
#
#
# for idx in list:
#     labelpath="/home/spi/outputeeg/test/"+idx+"/C3/label and fc_out.npz"
#     l_data = np.load(labelpath)
#     label_outs = l_data["label_outs"]
#     subpath=trainvideopath+idx
#     sublist=os.listdir(subpath)
#     l=len(sublist)
#     z=int(l*1)
#     for i in sublist:
#        n=int(i.split(".")[0])
#        l_data = np.load(labelpath)
#        label_outs = l_data["label_outs"]
#        source=idx+"/"+i
#        if n<len(label_outs):
#           dayin=int(label_outs[n])
#           with open("/home/spi/t1rainlist.txt" , 'a') as f:
#              f.write( source+" "+str(dayin)+ "\n")