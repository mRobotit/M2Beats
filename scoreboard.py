import os
import numpy as np
import mir_eval

"""
评价正确预测的命中窗口为 数据集两拍之间的距离的0.17(threshold)
节拍以秒为单位
x是人标注的节拍(数据集节拍)
y是节拍跟踪算法生成的节拍
"""
fps = 60
x = []
# gWA_sBM_c01_d25_mWA0_ch10
x.append([  
    # 0 lyx
    np.array([57,87,152,174,241,259,342,363,394,452,517,538,622])
    # 1 zyc
    ,np.array([83,174,265,367,455,537,625])
    # 2 wyl
    ,np.array([41,64,90,173,257,371,449,533,611,682])
    # 3 jdx
    ,np.array([83,171,260,363,451,536,622])
    # 4 chr
    ,np.array([72, 160, 245, 352, 442,526,616])
    # 5 database
    ,np.array([54, 98, 128, 174, 222, 278, 300, 409, 465, 487, 540, 589, 640, 667])
    # 6 tradition
    ,np.array([  6,  14,  23,  35,  45,  54,  60,  84,  93,  98, 112, 122, 133,
    141, 151, 173, 181, 199, 213, 231, 240, 245, 259, 269, 293, 302,
    314, 329, 350, 359, 368, 376, 381, 400, 413, 424, 435, 447, 451,
    460, 472, 481, 490, 504, 515, 524, 537, 546, 556, 566, 573, 578,
    583, 595, 601, 608, 622, 631, 648, 669, 683, 695, 700, 717])
    # 7 music
    ,np.array([  0.69659865,  45.27891   ,  89.86123   , 135.14014   ,
       181.11565   , 226.39456   , 270.28027   , 315.55917   ,
       360.1415    , 405.4204    , 450.00272   , 495.28165   ,
       539.86395   , 584.4463    , 629.72516   , 675.7007    ], int)
])

# gLO_sBM_c01_d14_mLO4_ch04
x.append([  
    # 0 lyx
    np.array([34, 93, 157, 215, 278,345,391,451])
    # 1 zyc
    ,np.array([44, 92, 156, 213, 272, 335, 391, 454])
    # 2 wyl
    ,np.array([60, 110 , 146, 181, 210, 276, 331, 395, 464])
    # 3 jdx
    ,np.array([32,89,153,211,269,331,387,450])
    # 4 chr
    ,np.array([32,101,214,275,390,447])
    # 5 database
    ,np.array([ 34,  90, 199, 258, 320, 391, 441])
    # 6 tradition
    ,np.array([  0,  14,  31,  42,  58,  71,  83,  91, 110, 124, 133, 143, 156,
    178, 191, 210, 227, 240, 253, 268, 297, 305, 312, 323, 332, 344,
    352, 360, 372, 381, 388, 410, 423, 434, 453, 466, 476])
    # 7 music
    ,np.array([  0.69659865,  29.953741  ,  60.604084  ,  90.55782   ,
       119.814964  , 150.4653    , 181.11565   , 209.6762    ,
       239.62993   , 270.28027   , 300.9306    , 330.18777   ,
       360.1415    , 390.09525   , 420.74557   , 450.00272   ,
       479.25986   ], int)
])

# gWA_sBM_c01_d27_mWA4_ch09
x.append([  
    # 0 lyx
    np.array([28, 55, 84, 116, 144,177,210,245,270,300,325,355,387,415,457])
    # 1 zyc
    ,np.array([21, 50, 86, 121, 149, 178, 210, 244, 268, 298, 328, 358, 391, 418, 447])
    # 2 wyl
    ,np.array([25, 48, 89, 124 , 146, 174, 193, 230 , 265 , 304 , 327 , 347 , 377, 407 , 433 , 460])
    # 3 jdx
    ,np.array([25,54,83,115,146,176,205,239,270,297,326,358,386,417,447])
    # 4 chr
    ,np.array([58,90,150,174,212,270,295,327,366,389,410,449])
    # 5 database
    ,np.array([ 11,  35,  93, 115, 144, 171, 202, 235, 268, 295, 324, 371, 392, 418, 442])
    # 6 tradition
    ,np.array([ 23,  42,  50,  75,  83, 114, 142, 162, 174, 191, 203, 231, 254,
    266, 274, 294, 316, 324, 340, 346, 355, 383, 405, 413, 435, 444,
    461, 468, 479])
    # 7 music
    ,np.array([  0.69659865,  29.953741  ,  59.907482  ,  90.55782   ,
       120.511566  , 150.4653    , 180.41905   , 210.37279   ,
       240.32652   , 270.28027   , 300.234     , 330.18777   ,
       360.1415    , 390.09525   , 420.04898   , 450.00272   ,
       479.25986   ], int)
])

# gLH_sBM_c01_d16_mLH3_ch07
x.append([  
    # 0 lyx
    np.array([67,92,137,154,206,225,270,285,339,352,400,415,464,479])
    # 1 zyc
    ,np.array([23,92,154,219,286,352,415,479])
    # 2 wyl
    ,np.array([21,45,79,113,144,174,214,246,299,339,369,403,432,466,496])
    # 3 jdx
    ,np.array([6,38,72,102,136,168,206,232,270,301,338,366,401,430,464,497])
    # 4 chr
    ,np.array([8,41,75,139,173,211,277,345,410,475])
    # 5 database
    ,np.array([28,51, 109, 135, 159, 205, 237, 270, 291, 315, 338, 368, 391, 414, 463, 484, 522])
    # 6 tradition
    ,np.array([  0,  15,  41,  53,  63,  71,  84, 106, 120, 124, 136, 150, 174,
    187, 194, 199, 215, 231, 238, 258, 268, 282, 295, 308, 320, 336,
    348, 364, 371, 374, 399, 413, 436, 449, 461, 474, 488, 496, 510,
    522])
    # 7 music
    ,np.array([  0.69659865,  33.436737  ,  50.1551    ,  65.48028   ,
        98.91701   , 131.65714   , 164.39728   , 197.13742   ,
       229.87756   , 262.61768   , 294.66122   , 312.07617   ,
       327.40137   , 360.8381    , 393.57822   , 426.31836   ,
       458.3619    , 491.10202   , 520.3592    ], int)
])

# gBR_sBM_c01_d04_mBR0_ch05
x.append([  
    # 0 lyx
    np.array([47, 82, 104, 224, 260, 283, 315, 362, 402, 437, 458, 498, 543, 586, 623, 643, 673])
    # 1 zyc
    ,np.array([49, 82, 104, 139, 182, 224, 260, 283, 315, 362, 402, 437, 458, 498, 543, 586, 623, 643, 673])
    # 2 wyl
    ,np.array([43 , 80 , 105 , 144, 175, 219, 257, 296,331,356 , 390, 429, 461, 496, 543, 574, 614, 645, 677, 706])
    # 3 jdx
    ,np.array([45,81,103,136,182,225,260,283,315,385,404,439,459,496,539,584,619,646,678])
    # 4 chr
    ,np.array([55,107,225,319,393,493,587,682])
    # 5 database
    ,np.array([ 32, 101, 132, 170, 224, 283, 315, 345 ,390, 457, 489, 522, 543, 566, 590, 641, 670, 711])
    # 6 tradition
    ,np.array([  5,  11,  21,  33,  43,  61,  66,  77,  87, 101, 114, 132, 154,
    168, 182, 200, 216, 223, 242, 256, 267, 281, 293, 311, 329, 346,
    361, 382, 390, 402, 414, 429, 441, 446, 458, 470, 491, 526, 540,
    547, 558, 569, 584, 599, 610, 613, 624, 631, 638, 672, 695, 713])
    # 7 music
    ,np.array([6.9659865e-01, 4.5278912e+01, 9.0557823e+01, 1.1284898e+02,
       1.3514014e+02, 1.5812790e+02, 1.8041905e+02, 2.2569797e+02,
       2.7028027e+02, 2.9257144e+02, 3.1555917e+02, 3.3785034e+02,
       3.6014151e+02, 3.8312924e+02, 4.0542041e+02, 4.5069931e+02,
       4.7299048e+02, 4.9528165e+02, 5.1826941e+02, 5.3986395e+02,
       5.6285168e+02, 5.8514288e+02, 6.3042175e+02, 6.7570068e+02,
       7.1819324e+02], int)
])



def counting_hit(x, y, threshold=0.17):
    count = 0

    idx_x = 0
    idx_y = 0
    left = 0
    right = 0
    while idx_y<len(y) and idx_x < len(x):
        if idx_x < len(x):
            right = x[idx_x]
            window = (right - left) * threshold
        else:
            right = x[-1] + window
        # 命中
        if right - window <= y[idx_y] <= right + window:
            count += 1
            left = right
            idx_y += 1
            idx_x += 1
        elif y[idx_y] < right-window:
            idx_y += 1
        else:
            left = right
            idx_x += 1
    return count

def precision_recall(x, y, threshold=0.17):
    """
    精确率：正确预测的节拍与全部预测节拍的比例  越高表示错误预测占比越少
    召回率：正确预测的节拍与数据集全部节拍的比例   越高表示踩中更多正确节拍
    """
    

    match_count = counting_hit(x,y,threshold)
    if len(y) == 0:
        precision = 0
    else:
        precision = float(match_count)/len(y)
    if len(x) == 0:
        recall = 0
    else:
        recall = float(match_count)/len(x)
    return precision,recall

# fmeasure 和 Cemgil 已经实现，不再赘述



def cul_score(x, y):
    x = x/60
    y = y / 60
    Fmeasure = mir_eval.beat.f_measure(x, y,0.14)
    pscore = mir_eval.beat.p_score(x,y,0.25)
    Cemgil = mir_eval.beat.cemgil(x, y,0.07)[0]
    CAMLct = mir_eval.beat.continuity(x, y, 0.175, 0.175)

    precision, recall = precision_recall(x,y,0.25)


    print('precision: {}, recall: {}'.format(round(precision, 3), round(recall,3)))
    print('P-score:', round(pscore, 3))
    print('F-measure:', round(Fmeasure, 3))
    print('Cemgil:', round(Cemgil, 3))
    print({m:v for m,v in zip(['CMLc', 'CMLt', 'AMLc', 'AMLt'], CAMLct)})

def main():
    """
    0 gWA_sBM_c01_d25_mWA0_ch10
    1 gLO_sBM_c01_d14_mLO4_ch04
    2 gWA_sBM_c01_d27_mWA4_ch09
    3 gLH_sBM_c01_d16_mLH3_ch07
    4 gBR_sBM_c01_d04_mBR0_ch05
    """
    

    """
    0 lyx
    1 zyc
    2 wyl
    3 jdx
    4 chr
    5 database
    6 tradion
    7 music
    """

    movie_name_list = ["gWA_sBM_c01_d25_mWA0_ch10", "gLO_sBM_c01_d14_mLO4_ch04",
"gWA_sBM_c01_d27_mWA4_ch09", "gLH_sBM_c01_d16_mLH3_ch07", "gBR_sBM_c01_d04_mBR0_ch05"]

    
    # Pscore_s = 0
    # precision_s = 0
    # recall_s = 0
    # for i in range(len(movie_name_list)):
    #     print('\n'+movie_name_list[i])
        
    #     #测试数据集
    #     # note_file_dir = "/data/jdx/code/mycode/train_data"
    #     # note_file = os.path.join(note_file_dir, movie_name_list[i] + '.npz')
    #     # fr = np.load(note_file, allow_pickle=True)
    #     # beat = fr['beat']
    #     # mask = fr['mask']
    #     # length = np.where(mask==1)[0][-1]
    #     # beat = beat[:length]
    #     # beat_pos = np.where(beat==1)[0]

    #     # 测试模型
    #     # inf_file_dir = "./output/inf/"
    #     # inf_file = os.path.join(inf_file_dir, movie_name_list[i]+'.npz')
    #     # fr = np.load(inf_file, allow_pickle=True)
    #     # beat_pos = fr['beat_pos']

    #     Pscore = mir_eval.beat.p_score(x[i][3], beat_pos,0.25)
    #     Pscore_s += Pscore
    #     precision, recall = precision_recall(x[i][3], beat_pos,0.25)
    #     precision_s += precision
    #     recall_s += recall
    # Pscore_s /= len(movie_name_list)
    # precision_s /= len(movie_name_list)
    # recall_s /= len(movie_name_list)

    # print("min score:\nPscore = {}\nprecision = {}\nrecall = {}"
    #       .format(Pscore_s,precision_s,recall_s))



    # 传统算法平均值
    Pscore_s = 0
    precision_s = 0
    recall_s = 0
    for i in range(len(movie_name_list)):
        print('\n'+movie_name_list[i])
        cul_score(x[i][0], x[i][5])

        Pscore = mir_eval.beat.p_score(x[i][0], x[i][5],0.25)
        Pscore_s += Pscore
        precision, recall = precision_recall(x[i][0], x[i][5],0.25)
        precision_s += precision
        recall_s += recall
    Pscore_s /= len(movie_name_list)
    precision_s /= len(movie_name_list)
    recall_s /= len(movie_name_list)

    print("min score:\nPscore = {}\nprecision = {}\nrecall = {}"
          .format(Pscore_s,precision_s,recall_s))


    # y = x[1]
    # for i in range(len(y)):
    #     y[i] = y[i] / 60

    # cul_score(y[3],y[6])



if __name__ == "__main__":
    main()

