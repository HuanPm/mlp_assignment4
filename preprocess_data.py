
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def circle(position,relation):
    order = ['E', 'S', 'W', 'N']
    idx=order.index(position)
    relation %= 4
    return order[idx+relation-4]

def write_event(filename, auc_pos, auc_len, auc, deal, vul):
    file_object=open(filename,'a')
    for i in range(auc_len):
        auc_to_i = auc_onehot(auc[:i])
        #auc_to_i = str(auc[:i])
        auc_i = auc_i_onehot(auc[i])
        #out_line = deal[circle(auc_pos,i)]+ vul+ auc_to_i + '\t' \
                               # +deal[circle(auc_pos,i+2)]+'\t'+auc_i + '\n'
        #print(i)
        out_line = deal[circle(auc_pos,i)]+vul+auc_to_i \
                                +deal[circle(auc_pos,i+2)] + auc_i
        out_line = ''.join([i+' ' for i in out_line])
        out_line += '\n'
        file_object.writelines(out_line)
    file_object.close()

def vul_onehot(vul_str):
    if 'None' in vul_str or vul_str == "":
        return '00'
    if 'EW' in vul_str:
        return '01'
    if 'NS' in vul_str:
        return '10'
    if 'Both' in vul_str or 'All' in vul_str:
        return '11'

def deal_onehot(deal_str):
    suit = deal_str[::-1].split('.')
    card = { '2':1,
             '3':2,
             '4':3,
             '5':4,
             '6':5,
             '7':6,
             '8':7,
             '9':8,
             'T':9,
             'J':10,
             'Q':11,
             'K':12,
             'A':13,
    }
    onehot = ''
    for i in range(4):
        onehot_suit = '0'*13
        for j in suit[i]:
            pos = card[j]-1
            onehot_suit = onehot_suit[:pos]+'1'+ onehot_suit[pos+1:]
        onehot += onehot_suit
    return onehot

def auc_onehot_openpass(auc):
    process_len = min(3,len(auc))
    open_pass = '0'*3
    openpass_len = 0
    for i in range(process_len):
        if auc[i] == 'Pass':
            open_pass = open_pass[:i]+'1'+open_pass[i+1:]
            openpass_len = i
        else:
            openpass_len = i
            break
    return open_pass,openpass_len

def auc_onehot_contrs(auc_contrs):
    leng = len(auc_contrs)
    contrs = '0'* 8
    non_contrs = {
        0:'Pass',
        1:'Pass',
        2:'X',
        3:'Pass',
        4:'Pass',
        5:'XX',
        6:'Pass',
        7:'Pass'
    }
    k = 0
    for i in range(leng-1):
        for j in range(k,8):
            if auc_contrs[i+1] == non_contrs[j]:
                contrs = contrs[:j] + '1' + contrs[j+1:]
                k = j+1
                break
    contrs = '1'+contrs
    return contrs

def auc_split(auc):
    _,openpass_len = auc_onehot_openpass(auc)

    auc_cut = auc[openpass_len:]
    auc_cut_len = len(auc_cut)
    split = []
    start = 0
    for i in range(1,auc_cut_len):
        if auc_cut[i][0].isdigit():
            split.append(list(auc_cut[start:i]))
            start = i
    if auc_cut[start:] != []:
        split.append(auc_cut[start:])
    return split

def auc_onehot(auc):
    auc_len = len(auc)
    open_pass,_ = auc_onehot_openpass(auc)
    split = auc_split(auc)
    level = [str(i) for i in range(1,8) for j in range(5)]
    suit = ['C', 'D', 'H', 'S', 'NT']*7
    suit_level = [level[i]+suit[i] for i in range(35)]
    j = 0
    onehot = open_pass
    for i in range(35):
        if j >= len(split):
            break
        if split[j][0] == suit_level[i]:
            onehot += auc_onehot_contrs(split[j])
            j += 1
        else:
            onehot += '0' * 9
    if len(onehot)< 318:
        onehot = onehot + '0'* (318-len(onehot))
    return onehot

def auc_i_onehot(auc_i):
    level=[str(i) for i in range(1,8) for j in range(5)]
    suit=['C','D','H','S','NT']*7
    suit_level=[level[i]+suit[i] for i in range(35)]
    suit_level.extend(['Pass', 'X', 'XX'])
    onehot = '0' * 38
    idx = suit_level.index(auc_i)
    onehot = onehot[:idx] + '1' +onehot[idx+1:]
    return onehot

def read_write_file(read_filename,write_filename):
    #print(write_filename)
    fp = open(read_filename)
    flag = None
    event = -1
    vul = []
    auc_lens = []
    auc_len = 0
    bid_types = []

    for line in fp:
        if line == '\n':
            continue
        content = line.split()
        if content[0] == '[Event':
            event += 1
            #print(event)
            deal = {}
            auc = []
            #auc = defaultdict(list)
            if event:
                auc_lens.append(auc_len)
            auc_len = 0

        if content[0] == '[Vulnerable':
            vul_str = content[1][1:-2]
            #print(vul_str)
            vul.append(vul_onehot(vul_str))
        if content[0] == '[Deal':
            pos = content[1][1]
            deal[pos] = deal_onehot(content[1][3:])
            deal[circle(pos,1)] = deal_onehot(content[2])
            deal[circle(pos,2)] = deal_onehot(content[3])
            deal[circle(pos,3)] = deal_onehot(content[4][:-2])
            # if content[1][1] == 'S':
            #     deal['S'] = content[1][3:]
            #     deal['W'] = content[2]
            #     deal['N'] = content[3]
            #     deal['E'] = content[4][:-2]
            # if content[1][1] == 'W':
            #     deal['W'] = content[1][3:]
            #     deal['N'] = content[2]
            #     deal['E'] = content[3]
            #     deal['S'] = content[4][:-2]
            # if content[1][1] == 'N':
            #     deal['N'] = content[1][3:]
            #     deal['E'] = content[2]
            #     deal['S'] = content[3]
            #     deal['W'] = content[4][:-2]
        if content[0]=='[Auction':
            if content[1] == '"E"]':
                flag = 'E'
            if content[1] == '"S"]':
                flag = 'S'
            if content[1] == '"W"]':
                flag = 'W'
            if content[1] == '"N"]':
                flag = 'N'
            continue
        # if content[0][0] == '[':

            # file_object=open("1996",'w')
            # if flag == 'E':
            #     file_object.writelines(deal['E']+'\t'+vul[event]+'\t'+str(auc['E'])+'\t'+deal['W']+'\n')
            #     file_object.writelines(str(deal['E'])+'\t'+str(vul[event])+'\t'+str(auc['E'])+'\t'+str(deal['W'])+'\n')
            #     file_object.writelines(str(deal['S'])+'\t'+str(vul[event])+'\t'+str(auc['S'])+'\t'+str(deal['N'])+'\n')
            #     file_object.writelines(str(deal['W'])+'\t'+str(vul[event])+'\t'+str(auc['W'])+'\t'+str(deal['E'])+'\n')
            #     file_object.writelines(str(deal['N'])+'\t'+str(vul[event])+'\t'+str(auc['N'])+'\t'+str(deal['S'])+'\n')
            # file_object.close()
            # write_event(write_filename,flag, auc_len, auc, deal, vul[event])
            # flag=None
            #break
        if flag != None:
            if content[0][0]=='[' or content == []:
                write_event(write_filename,flag,auc_len,auc,deal,vul[event])
                flag=None

        if flag != None:
            level=[str(i) for i in range(1,8) for j in range(5)]
            suit=['C','D','H','S','NT']*7
            suit_level=[level[i]+suit[i] for i in range(35)]
            suit_level.extend(['Pass','X','XX'])
            if content[:3] == ['Pass'] * 4:
                flag = None
                continue
            if content[-1] == 'AP':
                content.extend(['Pass']*3)
            dlt = []
            for i in content:
                if i not in suit_level:
                    dlt.append(i)
            content = [i for i in content if i not in dlt]
            length = len(content)
            #print(content)
            for i in range(6):
                if length > i:
                    auc.append(content[i])
                    auc_len += 1
                    bid_types.append(content[i])
                else:
                    break
            # if length > 0:
            #     auc.append(content[0])
            #     auc_len += 1
            # if length > 1:
            #     auc.append(content[1])
            #     auc_len += 1
            # if length > 2:
            #     auc.append(content[2])
            #     auc_len += 1
            # if length > 3:
            #     auc.append(content[3])
            #     auc_len += 1
            # if length > 4:
            #     auc.append(content[4])
            #     auc_len += 1
            # if length > 5:
            #     auc.append(content[5])
            #     auc_len += 1
        # if flag == 'S':
        #     length = len(content)
        #     auc['S'].append(content[0])
        #     if length > 1:
        #         auc['W'].append(content[1])
        #     if length > 2:
        #         auc['N'].append(content[2])
        #     if length > 3:
        #         auc['E'].append(content[3])
        # if flag == 'W':
        #     length = len(content)
        #     auc['W'].append(content[0])
        #     if length > 1:
        #         auc['N'].append(content[1])
        #     if length > 2:
        #         auc['E'].append(content[2])
        #     if length > 3:
        #         auc['S'].append(content[3])
        # if flag == 'N':
        #     length = len(content)
        #     auc['N'].append(content[0])
        #     if length > 1:
        #         auc['E'].append(content[1])
        #     if length > 2:
        #         auc['S'].append(content[2])
        #     if length > 3:
        #         auc['W'].append(content[3])
    return auc_lens,bid_types

def bid_len_plot(auc_lens):
    bar = []
    bar.append(len([i for i in auc_lens if 4<=i<=5])/len(auc_lens)*100)
    bar.append(len([i for i in auc_lens if 6<=i<=10])/len(auc_lens)*100)
    bar.append(len([i for i in auc_lens if 11<=i<=15])/len(auc_lens)*100)
    bar.append(len([i for i in auc_lens if 16<=i<=20])/len(auc_lens)*100)
    bar.append(len([i for i in auc_lens if 21<=i<=25])/len(auc_lens)*100)
    bar.append(len([i for i in auc_lens if 26<=i<=42])/len(auc_lens)*100)

    #print(bar)
    plt.figure()
    plt.bar([1,2,3,4,5,6],bar, align='center', color = '#00008B' )
    plt.xticks(np.arange(1,7),["4-5", "6-10", "11-15", "16-20", "21-25", "26-42"])

    plt.xlabel('Bidding length')
    plt.ylabel('Percentage(%)')
    plt.title("Bidding length distribution")


    bar_label = ["{:.2f}".format(i) for i in bar]
    for x, y, label in zip([1,2,3,4,5,6],bar, bar_label):
        plt.text(x,y,label,ha='center',va='bottom')

    # plt.xlim(4,42)

def bid_type_plot(bid_types):
    level=[str(i) for i in range(1,8) for j in range(5)]
    suit=['C','D','H','S','NT']*7
    suit_level=[level[i]+suit[i] for i in range(35)]
    suit_level.extend(['Pass','X','XX'])

    bar = []
    for i in range(38):
        bar.append(len([j for j in bid_types if j == suit_level[i]])/len(bid_types)*100)

    plt.figure()
    plt.bar([i+1 for i in range(38)], bar, align='center')
    plt.xticks(np.arange(1,39), [str(i) for i in range(38)])

    plt.xlabel('Bidding types')
    plt.ylabel('Percentage(%)')
    plt.title("Bidding types distribution")

    bar_label = ["{:.1f}".format(i) for i in bar]
    for x, y, label in zip([i+1 for i in range(38)], bar, bar_label):
        plt.text(x, y, label, ha = 'center', va = 'bottom')



#read_write_file('Dutch Teams_'+'1996'+'_20200314094348','1996_test')
data_year = ['1996','1997','1998','1999','2000','2001','2003','2004','2005',
             '2006','2009','2010','2011','2012','2013','2014','2015','2016','2017']
all_auc_lens = []
all_bid_types = []

for year in data_year:
    auc_lens,bid_types = read_write_file('Dutch Teams_'+year+'_20200314094348','all_year2')
    all_auc_lens += auc_lens
    all_bid_types += bid_types

bid_len_plot(all_auc_lens)
bid_type_plot(all_bid_types)

print(len(all_bid_types))

plt.show()


#read_write_file('Dutch Teams_'+'1999'+'_20200314094348','1999_space')