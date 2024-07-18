import os

def writelog(log,printflag = False):
    f = open('./log','a+')
    f.write(log+'\n')
    if printflag:
        print(log)

def show_paramsnumber(net):
    writelog('net parameters:'+str(sum(param.numel() for param in net.parameters())),True)
