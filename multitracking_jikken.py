import numpy as np
from numpy import linalg as LA
import cv2 as cv
import sys
import pandas as pd
import time
import socket
from scipy.optimize import fsolve

def transfer(x, real_dist):
    if x == 0:
        #k=0ならば640[px]:実測距離[m]=1[px]:1ピクセルの時の距離[m]
        return real_dist/640
    elif x == 1:
        #k=1ならば480[px]:実測距離[m]=1[px]:1ピクセルの時の距離[m]
        return real_dist/480
    else:
        return 1

#黒色を検出する関数
#歩行者群を検出する関数
def brack_detect_group(img, cliplimit, tilegrid, v, gauss, sigma, kernel_dilate, kernel_erode, area_min, area_max):

    """
    #ヒストグラム平滑化を行い，白飛びをなくす
    #フレームを各チャンネル毎，RGB毎に分けて，平滑化を行う
    #ヒストグラムの山をなだらかにする
    r, g, b = cv.split(img)
    clahe = cv.createCLAHE(cliplimit = cliplimit, tileGridSize = (tilegrid, tilegrid))
    r = clahe.apply(r)
    g = clahe.apply(g)
    b = clahe.apply(b)
    #分割したチャンネルを元に戻す
    img = cv.merge((r, g, b))
    """

    #ガウシアンフィルタ
    img = cv.GaussianBlur(img, ksize = (gauss, gauss), sigmaX = sigma, sigmaY = sigma)

    #RGB空間をHSV変換する
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    #inRange関数で範囲指定して二値化 ⇒ 白い領域を使用する
    mask_image = cv.inRange(hsv, (0, 0, 0), (179, 255, v))

    #膨張処理を行い，白い部分を大きくする
    mask_image = cv.dilate(mask_image, kernel_dilate, iterations = 1)

    #収縮処理を行い，白い部分を小さくする
    mask_image = cv.erode(mask_image, kernel_erode, iterations = 1)

    #ラベリング処理を行う
    num_labels, label_image, stats, center_pre = cv.connectedComponentsWithStats(mask_image)

    #ラベリング処理を行うと背景を数えてしまうので背景削除
    num_labels = num_labels - 1
    stats = np.delete(stats, 0, 0)
    center_pre = np.delete(center_pre, 0, 0)
    kari = num_labels

    #ブロブ面積が小さいやつを排除するために取得したindexを入れるリストを準備
    index_list = []

    #ブロブ面積がarea_min以下の場合はリストから削除
    #ブロブ面積がarea_max以上の場合はリストから削除
    for index in range(kari):
        menseki = stats[index][4]
        if menseki < area_min:
            index_list.append(index)
            num_labels = num_labels - 1
        if menseki > area_max:
            index_list.append(index)
            num_labels = num_labels - 1

    stats = np.delete(stats, index_list,0)
    center_pre = np.delete(center_pre, index_list, 0)

    #面積を排除するコードを記入
    stats = np.delete(stats, 4, 1)

    #trackingする際のcenter座標を格納するための配列を作成
    center = np.zeros((num_labels,2))

    #statsはtuple配列にしないとtracker.addができない
    stats = tuple(map(tuple, stats))

    return num_labels, stats, center, center_pre

#検出したオブジェクトを追跡するため初期化するための関数
#歩行者群追跡にはMOSSEを用いる
def tracking_group(tracker, image, num_groups, stats_group):
    # 検出した個数分繰り返し処理を行い，検出したオブジェクトにtrackerを作成
    for index in range(num_groups):
        tra = tracker.add(cv.TrackerMOSSE_create(), image, stats_group[index])

#初期化を行う関数
def init():
    init_group_person = False
    tracker = cv.MultiTracker_create()
    num = 0
    frame = 0
    count = 0

    return init_group_person, tracker, num, frame, count

#マーカの位置情報を抽出して，マーカを描画する関数
def marker_detect(marker_cen, aruco, dictionary, image, clipLimit, tilegrid):
    
    #ヒストグラム平滑化を行い，白飛びをなくす
    #ARマーカの4隅，IDを検出
    #フレームをRGBにわける，チャンネルごとに
    r, g, b = cv.split(image)
    clahe = cv.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilegrid,tilegrid))
    #ヒストグラム平滑化を行う
    r = clahe.apply(r)
    g = clahe.apply(g)
    b = clahe.apply(b)

    #平滑化処理したチャンネルを元に戻す
    image = cv.merge((r,g,b))

    #マーカを検出
    #マーカの4隅を検出
    corners, ids, rejectedImgPoints = aruco.detectMarkers(image, dictionary) #マーカを検出
    corners = np.array(corners)

    #4隅のｘ，ｙ座標の平均を取ってマーカ中心座標を決定
    if len(corners)>0:
        marker_x = 0
        marker_y = 0
        for x in range(4):
            marker_x += corners[0][0][x][0]
            marker_y += corners[0][0][x][1]
        marker_x = marker_x / 4
        marker_y = marker_y / 4
        marker_cen = np.array([int(marker_x), int(marker_y)])
        return marker_cen
    else:
        #何も検出されなかったらNoneを返す
        return np.array([None, None])

#更新したtrackerで追跡矩形の中心座標を取得
#更新したtrackerで歩行者群の楕円のwidth，heightを取得
#更新したtrackerで歩行者の円のwidth, heightを取得
def tracking_update(boxes):
    #trackerを更新して，矩形の中心座標を取得
    center_x = boxes[:,0] + boxes[:,2]/2
    center_y = boxes[:,1] + boxes[:,3]/2

    #歩行者群の楕円のwidth，heightを取得
    width = boxes[:,2]
    height = boxes[:,3]

    #num_labels行2列の2次元配列を作成．1列目に中心のx座標，2列目に中心のy座標
    center = np.r_["1,2,0", center_x, center_y]
    width_height = np.r_["1,2,0", width, height]

    return center, width_height

#目的地方向と歩行者群進行方向の角度差を出力
#x軸から時計回りを正とした時の歩行者群楕円角度を出力
def cal_theta_group(num_groups, center_group, center_pre_group, vector_goal):
    theta_group = np.zeros((num_groups))
    theta_ellipse = np.zeros((num_groups))
    vector_e = np.zeros((num_groups, 2))
    vector_cen = np.array(center_group - center_pre_group)

    #方向ベクトルからなす角の計算を行う
    for index in range(num_groups):
        i = np.inner(vector_goal, vector_cen[index])
        n = LA.norm(vector_goal)*LA.norm(vector_cen[index])
        if n == 0:
            theta = 0
        else:
            c = i / n
            theta = np.arccos(np.clip(c,-1.0,1.0))
        theta_group[index] = theta

    #基準であるｘ軸からの楕円の回転角度を求める
    #時計回りを正とする
    for index in range(num_groups):
        n = LA.norm(vector_cen[index])
        if n==0:
            theta=0
        else:
            #arctan2では-180°～180°まで計算可能
            theta = np.arctan2(vector_cen[index][1], vector_cen[index][0])
            theta = theta * 180 / np.pi
        theta_ellipse[index] = theta

    #進行方向とは逆の単位ベクトルを作成
    for index in range(num_groups):
        n = LA.norm(vector_cen[index])
        if n == 0:
            vector_e[index] = [-1,0]
        else:
            vector_e[index] = - (vector_cen[index] / n)
        
    return theta_group, theta_ellipse, vector_e

#描画処理を行う関数
#歩行者•歩行者群進行方向を描画
def drawing_vector(image, num_labels, center, center_pre):
    for line in range(num_labels):
        cv.arrowedLine(image, (int(center_pre[line,0]),int(center_pre[line,1])),(int(center[line,0]),int(center[line,1])),(0,250,0),2)

#描画処理を行う関数
#歩行者群位置，マーカ位置を描画
def all_drawing_detection_group(image, boxes, marker_cen, theta_ellipse):

    #マーカーの中心点
    marker_cen = tuple(marker_cen)

    #検出マーカーの中心描画，検出領域も描画した方がいいか？
    #マーカーの色も変えたほうが良い
    cv.circle(image,marker_cen, radius=3,color=(250,0,0),thickness=-1,lineType=cv.LINE_4)

    for i,newbox in enumerate(boxes):
        #矩形の左上頂点
        left_top = (int(newbox[0]), int(newbox[1]))
        #矩形の右下頂点
        right_bottom = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #楕円のwidth，height
        ellipse_width_height = (int(max(newbox[2]/2,newbox[3]/2)),int(min(newbox[2]/2,newbox[3]/2)))
        #検出領域の中心点
        cen = (int(newbox[0]+newbox[2]/2), int(newbox[1]+newbox[3]/2))

        #検出領域を矩形描画
        #cv.rectangle(image, left_top, right_bottom, (250,0,0),2)
        #検出領域を楕円描画
        cv.ellipse(image, cen, ellipse_width_height,angle=theta_ellipse[i],startAngle=0,endAngle=360,color=(250,0,0),thickness=2)
        #検出領域の中心描画
        cv.circle(image, cen,radius=4, color=(0,0,250),thickness=-1, lineType=cv.LINE_4)

#描画処理を行う関数
#歩行者群位置，マーカ位置を描画
def drawing_detection_group(image, boxes, marker_cen):

    #マーカーの中心点
    marker_cen = tuple(marker_cen)

    ##検出マーカーの中心描画，検出領域も描画した方がいいか？
    ##マーカーの色も変えたほうが良い
    cv.circle(image,marker_cen, radius=3,color=(250,0,0),thickness=-1,lineType=cv.LINE_4)

    for newbox in boxes:
        #矩形の左上頂点
        left_top = (int(newbox[0]), int(newbox[1]))
        #矩形の右下頂点
        right_bottom = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #楕円のwidth，height
        ellipse_width_height = (int(max(newbox[2]/2,newbox[3]/2)),int(min(newbox[2]/2,newbox[3]/2)))
        #検出領域の中心点
        cen = (int(newbox[0]+newbox[2]/2), int(newbox[1]+newbox[3]/2))

        #検出領域を矩形描画
        #cv.rectangle(image, left_top, right_bottom, (0,0,200),2)
        #検出領域の中心描画
        cv.circle(image, cen,radius=4, color=(0,0,250),thickness=-1, lineType=cv.LINE_4)

#描画処理を行う関数
#歩行者群位置は描画あり，マーカ位置は描画なし
def all_drawing_detection_without_marker_group(image, boxes,theta_ellipse):

    for i,newbox in enumerate(boxes):
        #矩形の左上頂点
        left_top = (int(newbox[0]), int(newbox[1]))
        #矩形の右下頂点
        right_bottom = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #楕円のwidth，height
        ellipse_width_height = (int(max(newbox[2]/2,newbox[3]/2)),int(min(newbox[2]/2,newbox[3]/2)))
        #検出領域の中心点
        cen = (int(newbox[0]+newbox[2]/2), int(newbox[1]+newbox[3]/2))

        #検出領域を矩形描画
        #cv.rectangle(image, left_top, right_bottom, (250,0,0),2)
        #検出領域の中心描画
        cv.circle(image, cen,radius=4, color=(0,0,250),thickness=-1, lineType=cv.LINE_4)
        #検出領域を楕円描画
        cv.ellipse(image, cen, ellipse_width_height,angle=theta_ellipse[i],startAngle=0,endAngle=360,color=(250,0,0),thickness=2)


#描画処理を行う関数
#歩行者群位置は描画あり，マーカ位置は描画なし
def drawing_detection_without_marker_group(image, boxes):

    for newbox in boxes:
        #矩形の左上頂点
        left_top = (int(newbox[0]), int(newbox[1]))
        #矩形の右下頂点
        right_bottom = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #楕円のwidth，height
        ellipse_width_height = (int(max(newbox[2]/2,newbox[3]/2)),int(min(newbox[2]/2,newbox[3]/2)))
        #検出領域の中心点
        cen = (int(newbox[0]+newbox[2]/2), int(newbox[1]+newbox[3]/2))

        #検出領域を矩形描画
        #cv.rectangle(image, left_top, right_bottom, (0,0,200),2)
        #検出領域の中心描画
        cv.circle(image, cen,radius=4, color=(0,0,250),thickness=-1, lineType=cv.LINE_4)

#ロボットと歩行者群の距離計算
def robot_group_dist(num_groups, marker_cen, center_group):

    group_dist = []

    for i in range(num_groups):
        group_dist.append(np.sqrt((marker_cen[0]-center_group[i][0])**2+(marker_cen[1]-center_group[i][1])**2))

    #ロボットのセンシング範囲内に存在しない歩行者群を除外
    for i in range(num_groups):
        if group_dist[i] > r_s:
            group_dist[i] = None
        else:
            pass

    return group_dist

#ロボットと各歩行者の距離計算
def robot_person_dist(num_persons, marker_cen, center_person):

    person_dist = []

    for i in range(num_persons):
        person_dist.append(np.sqrt((marker_cen[0]-center_person[i][0])**2+(marker_cen[1]-center_person[i][1])**2))

    #歩行者群と歩行者で同じプログラムであるからgoryu_joukenで分けたほうがいい
    #合流したらロボットより後ろの歩行者は認識しない
    #人間と同様に後ろは気にしない
    if goryu_jouken:
        for i in range(num_persons):
            if person_dist[i] > r_s or center_person[i][0] < marker_cen[0]:
                person_dist[i] = None
            else:
                pass
    else:
        for i in range(num_persons):
            if person_dist[i] > r_s:
                person_dist[i] = None
            else:
                pass
    return person_dist

#合流地点生成
def cal_goryu(num_groups, marker_cen, center_group, width_height, theta_group, theta_ellipse, vector_e, pedestrian_dist_mean):
    goryus = np.zeros((num_groups,2))
    goryu = []
    tmp = 0
    dist = 0
    ellipse = 0

    for index in range(num_groups):
        #目的地方向と異なる方向に進む歩行者群には合流地点を生成しない
        if theta_group[index] < np.pi/2 and theta_group[index] >= 0:
            if width_height[index][0] > width_height[index][1]:
                goryus[index] = center_group[index] + (width_height[index][0] / 2 + 0) * vector_e[index]

                #合流地点とロボットの距離を求める
                #その距離がセンシング範囲外だったらNoneを返す
                tmp = np.sqrt((marker_cen[0]-goryus[index][0])**2+(marker_cen[1]-goryus[index][1])**2)

                if tmp > r_s:
                    pass
                else:
                    #ロボットと合流地点の距離が一番短い合流地点を出力
                    if len(goryu) == 0:
                        dist = tmp
                        goryu = goryus[index]
                    elif dist > tmp:
                        dist = tmp
                        goryu = goryus[index]
                    else:
                        pass
            else:
                goryus[index] = center_group[index] + (width_height[index][1] / 2 + 0) * vector_e[index]

                #合流地点とロボットの距離を求める
                #その距離がセンシング範囲外だったらNoneを返す
                tmp = np.sqrt((marker_cen[0]-goryus[index][0])**2+(marker_cen[1]-goryus[index][1])**2)

                if tmp > r_s:
                    pass
                else:
                    #ロボットと合流地点の距離が一番短い合流地点を出力
                    if len(goryu) == 0:
                        dist = tmp
                        goryu = goryus[index]
                    elif dist > tmp:
                        dist = tmp
                        goryu = goryus[index]
                    else:
                        pass
        else:
            pass

    #歩行者群が全て90°以上，合流地点との距離がセンシング範囲外であるときはNoneを返す
    if len(goryu) == 0:
        goryu = [None,None]
        return goryu
    else:
        return goryu

#合流地点を描画
def drawing_goryu(image, goryu):
    #goryuがNoneであれば何も表示しない
    if goryu[0] is None or goryu[1] is None:
        pass
    else:
        goryu = (int(goryu[0]),int(goryu[1]))
        #合流地点を描画
        ##合流地点の色を変えたほうが良いかも？
        cv.circle(image, goryu,radius=4, color=(0,250,0),thickness=-1, lineType=cv.LINE_4)

#歩行者間距離を求め，sigma_ljを計算する関数
def cal_sigma_lj_group(num_persons, center_person, person_dist, r_person, r_person_mean):
    pedestrian_dist = []
    tmp_list = []
    tmp = 0
    sigma_lj = 0.6
    sigma_lj_tmp = 0
    pedestrian_dist_mean = 0

    #歩行者同士の距離を求め，それが指定の値より大きい場合は欠損値とする
    #さらに障害物の距離がNoneの場合は，欠損値とする
    for i in range(num_persons):
        for j in range(num_persons):
            if i == j:
                pass
            elif person_dist[i] == None or person_dist[j] == None:
                tmp_list.append(np.nan)
            else:
                tmp = np.sqrt((center_person[i][0]-center_person[j][0])**2 + (center_person[i][1]-center_person[j][1])**2)
                if (tmp - r_person[i] - r_person[j]) > person_dist_max:#歩行者と歩行者の距離から各歩行者半径を引いた接近距離で考える
                    tmp_list.append(np.nan)
                else:
                    tmp_list.append(tmp)
        pedestrian_dist.append(tmp_list)
        tmp_list = []

    #歩行者同士の距離平均を取って，歩行者間の平均距離を求める
    print("pedest_dist")
    print(pedestrian_dist)
    pedestrian_dist_mean = np.nanmean(pedestrian_dist)

    if np.isnan(pedestrian_dist_mean):
        #平均距離が求められなかったら，歩行者とロボットがギリギリ接しないロボットと歩行者の半径の和を離す
        return sigma_lj, r_ro + r_person_mean
    else:
        #合流するまではsigma_ljは一定とする
        return sigma_lj, pedestrian_dist_mean

#条件を切り替える関数
def condition_change_group(marker_cen, group_dist, goryu):
    global lj_jouken_group, lj_jouken_pedestrian, goryu_jouken, goal_jouken, init_person
    
    #歩行者群に対して
    #LJポテンシャルを発生させるか，させないかの条件
    #センシング範囲内に歩行者群があればLJポテンシャルを発生
    #なければゴールに引力ポテンシャルを発生
    if any(group_dist):
        lj_jouken_group = True
    else:
        lj_jouken_group = False

    #歩行者群の後方に合流できたら，合流地点に発生するポテンシャルをなくす
    if goryu[0] is None or goryu[1] is None:
        pass
    elif np.sqrt((marker_cen[0]-goryu[0])**2+(marker_cen[1]-goryu[1])**2) < goryu_dist:
        goryu_jouken = True
    else:
        pass
    print("goal")
    print(np.sqrt((marker_cen[0]-goal[0])**2+(marker_cen[1]-goal[1])**2))
    #ロボットのセンシング範囲内に目的地が入れば，目的地にだけ引力を生成
    if np.sqrt((marker_cen[0]-goal[0])**2+(marker_cen[1]-goal[1])**2) < r_s:
        goal_jouken = True

#LJポテンシャル計算
#歩行者群に対してLJポテンシャルを生成
def cal_pot_group(marker_cen_x, marker_cen_y, num_groups, center_group, theta_group, sigma_lj, goryu, r_area):
    tmp_pot = 0
    pot_all = 0

    #ロボット位置座標で障害物，歩行者の距離が変化するためcal_potで計算する必要あり
    group_dist = robot_group_dist(num_groups, [marker_cen_x,marker_cen_y], center_group)

    if lj_jouken_group:
        if goal_jouken:
            tmp_pot = -1/np.sqrt((marker_cen_x-goal[0])**2+(marker_cen_y-goal[1])**2)
            pot_all += tmp_pot * weight_goal
        else:
            for i in range(num_groups):
                if group_dist[i] is None:
                    tmp_pot = 0
                    pot_all += tmp_pot
                else:
                    a,b = weight(theta_group[i])
                    r = group_dist[i]
                    tmp_pot = 4*e_lj*(b*((sigma_lj+s*(1-a))/(r-(r_ro+r_area[i])))**p-a*((sigma_lj+s*(1-a))/(r-(r_ro+r_area[i])))**q)
                    print(tmp_pot)
                    print(a,b)
                    print(r)
                    pot_all += tmp_pot

            if goryu[0] is None or goryu[1] is None:
                tmp_pot =0
                pot_all += tmp_pot
            else:
                tmp_pot = -1/np.sqrt((marker_cen_x-goryu[0])**2+(marker_cen_y-goryu[1])**2)
                tmp_pot = tmp_pot * weight_goal
                pot_all += tmp_pot

    else:
        pot_all = -1/np.sqrt((marker_cen_x-goal[0])**2+(marker_cen_y-goal[1])**2)
        pot_all = pot_all * weight_goal

    return pot_all

#wrapped normal distribution関数
#進行方向の角度差で重み付けを行う
def f(theta):
    sum = 0
    for i in range(-10,11):
        a = np.exp(-(theta+2*np.pi*i)**2/(2*sigma_wn_2))
        sum += a

    return (1/(np.sqrt(2*np.pi*sigma_wn_2)))*sum

def weight(theta):
    a = f(theta)/f(0)
    b = (1-w)*a+w
    return a,b

#歩行者群の場合
#速度計算
def cal_velocity_group(marker_cen, num_groups, num_persons, center_group, center_pre_group, width_height, center_person, r_person, r_person_mean):
    
    #ピクセル座標を実座標に変換
    #だが，計算の部分のみ変換するだけで，他全部を変えてしまうと，描画ができなくなるので計算のみ変換
    marker_cen = marker_cen * transfer(x, real_dist)
    center_group = center_group * transfer(x, real_dist)
    center_pre_group = center_pre_group * transfer(x, real_dist)
    width_height = width_height * transfer(x, real_dist)
    center_person = center_person * transfer(x, real_dist)
    r_person = r_person * transfer(x, real_dist)
    r_person_mean = r_person_mean * transfer(x, real_dist)

    #歩行者群の進行方向角度差，y軸からの楕円角度を取得
    theta_group, theta_ellipse, vector_e = cal_theta_group(num_groups, center_group, center_pre_group, vector_goal)

    #ここから下がLJポテンシャル生成
    #歩行者群距離
    group_dist = robot_group_dist(num_groups, marker_cen, center_group)

    #歩行者距離
    person_dist = robot_person_dist(num_persons, marker_cen, center_person)
    print("person_dist")
    print(person_dist)

    #sigma_ljを出力
    sigma_lj, pedestrian_dist_mean = cal_sigma_lj_group(num_persons, center_person, person_dist, r_person, r_person_mean)

    #合流地点生成
    goryu = cal_goryu(num_groups, marker_cen, center_group, width_height, theta_group, theta_ellipse, vector_e, pedestrian_dist_mean)
    print("goryu")
    print(goryu)

    #歩行者•歩行者群の中心から円周までの距離を求める
    #ロボットのセンシング範囲外だったらどうするか
    #ポテンシャル計算でセンシング範囲外だったら除外するので考えない
    r_area = cal_r_area_group(marker_cen[0], marker_cen[1], num_groups, center_group, width_height, theta_ellipse)

    #条件変更
    condition_change_group(marker_cen, group_dist, goryu)
    
    #LJポテンシャル計算
    vx = -(cal_pot_group(marker_cen[0]+delt, marker_cen[1], num_groups, center_group, theta_group, sigma_lj, goryu, r_area)-cal_pot_group(marker_cen[0], marker_cen[1], num_groups, center_group, theta_group, sigma_lj, goryu, r_area))
    vy = -(cal_pot_group(marker_cen[0], marker_cen[1]+delt, num_groups, center_group, theta_group, sigma_lj, goryu, r_area)-cal_pot_group(marker_cen[0], marker_cen[1], num_groups, center_group, theta_group, sigma_lj, goryu, r_area))
    v = np.sqrt(vx**2+vy**2)

    #速度を正規化
    vx /= v/v_max
    vy /= v/v_max

    #goryuを描画するためにピクセル座標に変換
    if goryu[0] == None or goryu[1] == None:
        pass
    else:
        goryu = goryu / transfer(x, real_dist)

    print("marker")
    print(marker_cen)
    print("group")
    print(center_group)
    print("pedestrian")
    print(center_person)
    print("width_height")
    print(width_height)
    print("r_area")
    print(r_area)
    print("pedestrian_mean")
    print(pedestrian_dist_mean)
    print(sigma_lj)
    print("r_person")
    print(r_person)
    print(vx,vy)

    return theta_group, theta_ellipse, vector_e, goryu, vx, vy

#TCP通信を行い，速度指令を出力
def tcp(vx,vy):

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #s.connect(("192.168.1.1",1900))
    s.connect((socket.gethostname(),52720))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    vx_vy = "MOVE"+","+str(vx)+","+str(vy)
    data = vx_vy.encode()
    s.send(data)

#傾いた楕円方程式
def func(start, marker_cen_x, marker_cen_y, center_x, center_y, width, height, theta_ellipse):
    x, y = start[0], start[1]
    f = np.zeros(2)

    #ロボットの中心と歩行者群の中心を結んだ直線の方程式
    f[0] = y - (marker_cen_y-center_y)/(marker_cen_x-center_x)*(x-center_x)-center_y
    #角度θ回転した場合
    #楕円角度を入力できるようにする
    f[1] = np.power(((x-center_x)*np.cos(theta_ellipse*np.pi/180)+(y-center_y)*np.sin(theta_ellipse*np.pi/180))/max(width,height),2)+np.power((-(x-center_x)*np.sin(theta_ellipse*np.pi/180)+(y-center_y)*np.cos(theta_ellipse*np.pi/180))/min(width,height),2)-1
    return f

#歩行者群の中心から円周までの距離
def cal_r_area_group(marker_cen_x, marker_cen_y, num_groups, center_group, width_height, theta_ellipse):
    cross = []
    tmp = 0
    r_area = np.zeros(num_groups)
    for i in range(num_groups):
        tmp = fsolve(func,[marker_cen_x, marker_cen_y],(marker_cen_x, marker_cen_y, center_group[i][0], center_group[i][1], width_height[i][0]/2, width_height[i][1]/2, theta_ellipse[i]))
        cross.append(tmp)
        r_area[i] = np.sqrt((cross[i][0]-center_group[i][0])**2+(cross[i][1]-center_group[i][1])**2)
        cen=(int(cross[i][0]),int(cross[i][1]))
        cv.circle(image,cen ,radius=4, color=(0,0,250),thickness=-1, lineType=cv.LINE_4)
    return r_area

#動画の処理速度fpsを計算する関数
def fps(tm):
    tm.stop()
    fps = max_count/tm.getTimeSec()
    tm.reset()
    tm.start()
    #cv.putText(image, text=str(fps), org=(20,100), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0,255,0), thickness=3)
    print("--------------------------------")
    print(fps)
    return 0

#歩行者半径
def cal_r_person(num_persons, stats_person):
    r_person = np.zeros(num_persons)
    for i in range(num_persons):
        r_person[i] = max(stats_person[i][2], stats_person[i][3])/2
    r_person_mean = np.mean(r_person)
    return r_person, r_person_mean

####################################################################

#歩行者を検出する関数
#歩行者間距離を求めるために使用
def brack_detect_person(img, v, cliplimit, tilegrid, gauss, sigma, kernel, area_min, area_max):

    """
    #ヒストグラム平滑化を行い，白飛びをなくす
    #フレームを各チャンネル毎，RGB毎に分けて，平滑化を行う
    #ヒストグラムの山をなだらかにする
    r, g, b = cv.split(img)
    clahe = cv.createCLAHE(cliplimit = cliplimit, tileGridSize = (tilegrid, tilegrid))
    r = clahe.apply(r)
    g = clahe.apply(g)
    b = clahe.apply(b)
    #分割したチャンネルを元に戻す
    img = cv.merge((r, g, b))
    """

    #ガウシアンフィルター
    img = cv.GaussianBlur(img, ksize=(gauss,gauss), sigmaX=sigma, sigmaY=sigma)

    #フレームをhsvやrgbに変換する
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    # inRange関数で範囲指定２値化 -> マスク画像として使う
    mask_image = cv.inRange(hsv, (0,0,0), (179,255,v))

    #モルフォロジー変換で穴を埋めて，細かいノイズを除去
    #クローズ処理，オープニング処理は一回でいい．何回やっても変わらない
    #適切な処理を行う必要あり
    #mask_image = cv.morphologyEx(mask_image, cv.MORPH_CLOSE, kernel)
    #mask_image = cv.morphologyEx(mask_image, cv.MORPH_OPEN, kernel)

    #ラベリング処理を行う
    num_labels, label_image, stats, center_pre = cv.connectedComponentsWithStats(mask_image)

    #ラベリング処理を行うと背景を数えてしまうので背景削除
    num_labels = num_labels - 1
    stats = np.delete(stats, 0, 0)
    center_pre = np.delete(center_pre, 0, 0)
    kari = num_labels

    #ブロブ面積が小さいやつを排除するために取得したindexを入れるリストを準備
    index_list = []

    #ブロブ面積がarea_min以下の場合はリストから削除
    #ブロブ面積がarea_max以上の場合はリストから削除
    for index in range(kari):
        menseki = stats[index][4]
        if menseki < area_min:
            index_list.append(index)
            num_labels = num_labels - 1
        if menseki > area_max:
            index_list.append(index)
            num_labels = num_labels - 1

    stats = np.delete(stats, index_list,0)
    center_pre = np.delete(center_pre, index_list, 0)

    #面積を排除するコードを記入
    stats = np.delete(stats, 4, 1)

    # trackingする際のcenter座標を格納するための配列を作成
    center = np.zeros((num_labels,2))

    # statsはtuple配列にしないとtracker.addができない
    stats = tuple(map(tuple, stats))

    return num_labels, stats, center_pre, center


#検出したオブジェクトを追跡するため初期化するための関数
#歩行者追跡にはKCFを用いる
def tracking_person(tracker, image, num_persons, stats_person):
    # 検出した個数分繰り返し処理を行い，検出したオブジェクトにtrackerを作成
    for index in range(num_persons):
        tra = tracker.add(cv.TrackerKCF_create(), image, stats_person[index])

#目的地方向と歩行者進行方向の角度差を出力
def cal_theta_person(num_persons, center_person, center_pre_person, vector_goal):
    theta_person = np.zeros((num_persons))
    vector_cen = np.array(center_person - center_pre_person)
    #方向ベクトルからなす角の計算を行う
    for index in range(num_persons):
        i = np.inner(vector_goal, vector_cen[index])
        n = LA.norm(vector_goal)*LA.norm(vector_cen[index])
        if n == 0:
            theta = 0
        else:
            c = i / n
            theta = np.arccos(np.clip(c,-1.0,1.0))
        theta_person[index] = theta

    return theta_person

#描画処理を行う関数
#歩行者位置，マーカ位置を描画
def drawing_detection_person(image, boxes, marker_cen):

    #マーカーの中心点
    marker_cen = tuple(marker_cen)

    #検出マーカーの中心描画，検出領域も描画した方がいいか？
    #マーカーの色も変えたほうが良い
    cv.circle(image,marker_cen, radius=3,color=(250,0,0),thickness=-1,lineType=cv.LINE_4)

    for newbox in boxes:
        #矩形の左上頂点
        left_top = (int(newbox[0]), int(newbox[1]))
        #矩形の右下頂点
        right_bottom = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #円のwidth, height
        circle_width_height = int(max(newbox[2]/2, newbox[3]/2))
        #検出領域の中心点
        cen = (int(newbox[0]+newbox[2]/2), int(newbox[1]+newbox[3]/2))

        #検出領域を矩形描画
        #cv.rectangle(image, left_top, right_bottom, (0,0,200),2)
        #検出領域を円描画
        cv.circle(image, cen, radius = circle_width_height, color=(250,0,0), thickness=2)
        #検出領域の中心描画
        cv.circle(image, cen,radius=4, color=(0,0,250),thickness=-1, lineType=cv.LINE_4)

#描画処理を行う関数
#歩行者位置は描画あり，マーカ位置は描画なし
def drawing_detection_without_marker_person(image, boxes):

    for newbox in boxes:
        #矩形の左上頂点
        left_top = (int(newbox[0]), int(newbox[1]))
        #矩形の右下頂点
        right_bottom = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #円のwidth, height
        circle_width_height = int(max(newbox[2]/2, newbox[3]/2))
        #検出領域の中心点
        cen = (int(newbox[0]+newbox[2]/2), int(newbox[1]+newbox[3]/2))

        #検出領域を矩形描画
        #cv.rectangle(image, left_top, right_bottom, (0,0,200),2)
        #検出領域を円描画
        cv.circle(image, cen, radius = circle_width_height, color=(250,0,0), thickness=2)
        #検出領域の中心描画
        cv.circle(image, cen,radius=4, color=(0,0,250),thickness=-1, lineType=cv.LINE_4)

        
#条件を切り替える関数
def condition_change_person(marker_cen, person_dist):
    global lj_jouken_group, lj_jouken_pedestrian, goryu_jouken, goal_jouken

    #各歩行者に対して
    #LJポテンシャルを発生させるか，させないかの条件
    #センシング範囲内に歩行者群があればLJポテンシャルを発生
    #なければゴールに引力ポテンシャルを発生
    if any(person_dist):
        lj_jouken_pedestrian = True
    else:
        lj_jouken_pedestrian = False
    print("goal")
    print(np.sqrt((marker_cen[0]-goal[0])**2+(marker_cen[1]-goal[1])**2))
    #ロボットのセンシング範囲内に目的地が入れば，目的地にだけ引力を生成
    if np.sqrt((marker_cen[0]-goal[0])**2+(marker_cen[1]-goal[1])**2) < r_s:
        goal_jouken = True

#LJポテンシャル計算
#歩行者に対してLJポテンシャルを生成
def cal_pot_person(marker_cen_x, marker_cen_y, num_persons, center_person, theta_person, sigma_lj, r_area_person):
    tmp_pot = 0
    pot_all = 0

    #ロボット位置座標で障害物，歩行者の距離が変化するためcal_potで計算する必要あり
    person_dist = robot_person_dist(num_persons,[marker_cen_x,marker_cen_y],center_person)

    if lj_jouken_pedestrian:
        if goal_jouken:
            tmp_pot = -1/np.sqrt((marker_cen_x-goal[0])**2+(marker_cen_y-goal[1])**2)
            pot_all += tmp_pot * weight_goal
        else:
            #歩行者にLJポテンシャルを生成したい
            for i in range(num_persons):
                if person_dist[i] is None:
                    tmp_pot = 0
                    pot_all += tmp_pot
                else:
                    a,b = weight(theta_person[i])
                    r = person_dist[i]
                    tmp_pot = 4*e_lj*(b*((sigma_lj+s*(1-a))/(r-(r_ro+r_area_person[i])))**p-a*((sigma_lj+s*(1-a))/(r-(r_ro+r_area_person[i])))**q)
                    print(tmp_pot)
                    pot_all += tmp_pot
    else:
        pot_all = -1/np.sqrt((marker_cen_x-goal[0])**2+(marker_cen_y-goal[1])**2)
        pot_all = pot_all * weight_goal

    return pot_all

#歩行者の場合
#速度計算
def cal_velocity_person(marker_cen, num_persons, center_person, center_pre_person, width_height):

    #ピクセル座標を実座標に変換
    #だが，計算の部分のみ変換するだけで，他全部を変えてしまうと，描画ができなくなるので計算のみ変換
    marker_cen = marker_cen * transfer(x, real_dist)
    width_height = width_height * transfer(x, real_dist)
    center_person = center_person * transfer(x, real_dist)
    center_pre_person = center_pre_person * transfer(x, real_dist)

    #歩行者の進行方向角度差
    theta_person = cal_theta_person(num_persons, center_person, center_pre_person, vector_goal)

    #ここから下がLJポテンシャル生成
    #歩行者距離
    person_dist = robot_person_dist(num_persons, marker_cen, center_person)

    #歩行者の中心から円周までの距離
    r_area_person = cal_r_area_person(num_persons, width_height)

    #sigma_ljを出力
    sigma_lj, pedestrian_dist_mean = cal_sigma_lj_person(num_persons, center_person, person_dist, r_area_person)

    #条件変更
    condition_change_person(marker_cen, person_dist)

    #LJポテンシャル計算
    vx = -(cal_pot_person(marker_cen[0]+delt, marker_cen[1], num_persons, center_person, theta_person, sigma_lj, r_area_person)-cal_pot_person(marker_cen[0], marker_cen[1], num_persons, center_person, theta_person, sigma_lj, r_area_person))
    vy = -(cal_pot_person(marker_cen[0], marker_cen[1]+delt, num_persons, center_person, theta_person, sigma_lj, r_area_person)-cal_pot_person(marker_cen[0], marker_cen[1], num_persons, center_person, theta_person, sigma_lj, r_area_person))
    v = np.sqrt(vx**2+vy**2)

    #速度を正規化
    vx /= v/v_max
    vy /= v/v_max

    print("person_dist")
    print(person_dist)
    print("marker")
    print(marker_cen)
    print("center_person")
    print(center_person)
    print("width_height")
    print(width_height)
    print("sigma_lj")
    print(sigma_lj)
    print("r_area")
    print(r_area_person)
    print("pedestrian_mean")
    print(pedestrian_dist_mean)
    print(vx,vy)

    return theta_person, vx, vy

#歩行者の中心から円周までの距離
def cal_r_area_person(num_persons, width_height):
    r_area = np.zeros(num_persons)
    for i in range(num_persons):
        r_area[i] = max(width_height[i][0], width_height[i][1])/2
    return r_area

#歩行者間距離を求め，sigma_ljを計算する関数
def cal_sigma_lj_person(num_persons, center_person, person_dist, r_area_person):
    pedestrian_dist = []
    tmp_list = []
    tmp = 0
    sigma_lj = 0.6
    sigma_lj_tmp = 0
    pedestrian_dist_mean = 0

    #歩行者の平均半径を求める
    r_area_person_mean = np.mean(r_area_person)

    #歩行者同士の距離を求め，それが指定の値より大きい場合は欠損値とする
    #さらに障害物の距離がNoneの場合は，欠損値とする
    for i in range(num_persons):
        for j in range(num_persons):
            if i == j:
                pass
            elif person_dist[i] == None or person_dist[j] == None:
                tmp_list.append(np.nan)
            else:
                tmp = np.sqrt((center_person[i][0]-center_person[j][0])**2 + (center_person[i][1]-center_person[j][1])**2)
                if (tmp - r_area_person[i] - r_area_person[j]) > person_dist_max:
                    tmp_list.append(np.nan)
                else:
                    tmp_list.append(tmp)
        pedestrian_dist.append(tmp_list)
        tmp_list = []

    #歩行者同士の距離平均を取って，歩行者間の平均距離を求める
    pedestrian_dist_mean = np.nanmean(pedestrian_dist)

    if np.isnan(pedestrian_dist_mean):
        #平均距離が求められなかったら，歩行者とロボットがギリギリ接しないロボットと歩行者の半径の和を離す
        #歩行者間距離が求められないことは混雑環境ではないと考えて，歩行者間の平均距離にする
        return sigma_lj, r_ro + r_area_person_mean
    else:
        #歩行者間の平均距離から，ロボットと歩行者の距離を決めるsigma_ljを求める
        sigma_lj_tmp = (pedestrian_dist_mean - (r_ro + r_area_person_mean)) / 2

        #歩行者流に合流したらsigma_ljを変化させる
        if sigma_lj < sigma_lj_tmp:
            return sigma_lj, pedestrian_dist_mean
        else:
            sigma_lj = sigma_lj_tmp
            return sigma_lj, pedestrian_dist_mean



#main関数

#imageを表示するためのwindowを作成
cv.namedWindow("OpenCV Window", cv.WINDOW_NORMAL)

#カメラの準備
camera = cv.VideoCapture(0,cv.CAP_DSHOW)

#Trackingを開始するための呼び出し
tracker = cv.MultiTracker_create()

#ここから下は変数の定義
num = 0  #100フレーム毎に初期化する→１からリセット
frame = 0  #10フレーム毎に方向ベクトルを出力→1フレーム毎だと進む方向がわからない
x = 1 #x=0：640px，x=1：480px，x=左以外：pxのまま出力
real_dist = 1.2 #[m]で，640，480に値する実際距離になる
marker_cen = np.array([0,0]) #ロボットの初期位置，マーカの初期化
vector_goal = np.array([10,0]) #目的地方向
goal = np.array([640,480]) * transfer(x, real_dist) #ゴール地点，ピクセルではなく実距離にする

#前処理変数_group_pedestrian
v = 30  #HSV抽出．Vで二値化するための明度の最大値を決める
cliplimit = 1 #ヒストグラム平滑化，クリップリミット，コントラストの制限
tilegrid = 1 #ヒストグラム平滑化，タイルグリッドのサイズ
gauss_group = 3  #ガウシアンフィルターのカーネルサイズ
sigma_group = 0 #ガウス分布→大きいほどぼやける，0だとカーネルサイズから自動で計算してくれる
kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (74,74))
kernel_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (74,74))
area_min_group = 7000 #抽出する最小面積
area_max_group = 400000  #抽出する最大面積

v = 30  #HSV抽出．Vで二値化するための明度の最大値を決める
cliplimit = 1 #ヒストグラム平滑化，クリップリミット，コントラストの制限
tilegrid = 1 #ヒストグラム平滑化，タイルグリッドのサイズ
gauss_person = 3
sigma_person = 0
kernel_person = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4,4))
area_min_person = 1000
area_max_person = 30000


#LJポテンシャル変数定義
r_s = 4 #センシング範囲
r_ro = 0.3 #ロボット半径
r_hu = 0.3 #歩行者半径
person_dist_max = 1.2 #これ以上離れた歩行者間距離は考慮しない．中心位置からではなく，接触距離とする．
goryu_dist = 0.1 #合流地点とロボット位置がどれぐらい近くなれば合流地点を消すかのパラメータ
weight_goal = 40
sigma_wn_2 = 1/5
p=2
q=1
w=7.0*1e-3
s=0
e_lj=2
delt = 1/3
v_max = 1.3

#条件
init_group = False 
init_person = False
lj_jouken_group = False
lj_jouken_pedestrian = False
goryu_jouken = False
goal_jouken = False

#arucoライブラリ
aruco = cv.aruco 
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

#fpsを計測するための準備
tm = cv.TickMeter()
tm.start()
count = 0
max_count =10

#歩行速度を計測するための準備
tv = cv.TickMeter()
tv.start()

#検出•追跡処理
while camera.isOpened():
    ok, image = camera.read()

    if not ok:
        print("no image to read")
        break

    #合流したときに追跡対象を変化させる
    #歩行者と歩行者群を同時に追跡できれば良いのだが，処理が重くなってしまうので切り替える
    if goryu_jouken:
        #init_personが負の時に実行
        #合流したら歩行者追跡を行う
        if not init_person:
            init_person, tracker, num, frame, count = init()
            num_persons, stats_person, center_pre_person, center_person = brack_detect_person(image, v, cliplimit, tilegrid, gauss_person, sigma_person, kernel_person, area_min_person, area_max_person)
            tracking_person(tracker, image, num_persons, stats_person)
            init_person = True

        #もし黒色が検出されなかったらループの最初に戻る
        #init_personをFalseにして再び黒色を検出する
        if num_persons == 0:
            print("Not detect brack color!")
            cv.imshow('OpenCV Window', image)
            init_person = False
            k = cv.waitKey(1)
            if k == 27 : 
                break # esc pressed
        elif num == 100000:
            init_person, tracker, num, frame, count = init()
        else:
            #trackerの更新を行う
            ok, boxes = tracker.update(image)

            #trackerの更新ができなかったら検出からやり直し
            if not ok:
                init_person, tracker, num, frame, count = init()
            else:
               #マーカーの検出
               marker_cen = marker_detect(marker_cen, aruco, dictionary, image, cliplimit, tilegrid)

               #追跡歩行者の中心座標，矩形のwidth_heightを取得
               center_person, width_height = tracking_update(boxes)

               #マーカーが検出できなかったら
               if marker_cen[0] is None or marker_cen[1] is None:
                    #数フレーム毎に歩行者の進行方向を出力
                    if frame == 2:
                    
                        #歩行者の進行方向角度差
                        theta_person = cal_theta_person(num_persons, center_person, center_pre_person, vector_goal)

                        #描画する関数
                        drawing_vector(image, num_persons, center_person, center_pre_person)
                        drawing_detection_without_marker_person(image, boxes)

                        #center_preを更新する
                        center_pre_person = center_person

                        #frameを初期化する
                        frame = 0

                        #処理結果を表示
                        cv.imshow("OpenCV Window", image)

                        #fps計算
                        if count == max_count:
                            count = fps(tm)

                        #カウントを行う
                        num += 1
                        frame += 1
                        count += 1

                        #escを押したらbreakする
                        k = cv.waitKey(1)
                        if k == 27 : break # esc pressed
                    else:
                        #描画する関数
                        drawing_detection_without_marker_person(image, boxes)

                        #処理結果を表示
                        cv.imshow("OpenCV Window", image)

                        #fps計算
                        if count == max_count:
                            count = fps(tm)

                        #カウントを行う
                        num += 1
                        frame += 1
                        count += 1

                        #escを押したらbreakする
                        k = cv.waitKey(1)
                        if k == 27 : break # esc pressed
               else:
                    #ロボットがゴールに到達したらfor文を抜ける
                    if np.sqrt((marker_cen[0]*transfer(x, real_dist)-goal[0])**2+(marker_cen[1]*transfer(x, real_dist)-goal[1])**2) < 0.1:
                        break 
                    #数フレーム毎に歩行者の進行方向を出力
                    elif frame == 2:
                        #LJポテンシャルを計算
                        theta_person, vx, vy = cal_velocity_person(marker_cen, num_persons, center_person, center_pre_person, width_height)

                        #tcp通信
                        #tcp(vx, vy)

                        #描画する関数
                        drawing_vector(image, num_persons, center_person, center_pre_person)
                        drawing_detection_person(image, boxes, marker_cen)

                        #center_preを更新する
                        ###この前でtcp通信を行い，速度も取得する．この部分を起点とするから誤差が少ない
                        center_pre_person = center_person

                        #frameを初期化する
                        frame = 0

                        #処理結果を表示
                        cv.imshow("OpenCV Window", image)

                        #fps計算
                        if count == max_count:
                            count = fps(tm)

                        #カウントを行う
                        num += 1
                        frame += 1
                        count += 1

                        #escを押したらbreakする
                        k = cv.waitKey(1)
                        if k == 27 : break # esc pressed
                    else:
                        #描画する関数
                        drawing_detection_person(image, boxes, marker_cen)

                        #処理結果を表示
                        cv.imshow("OpenCV Window", image)

                        #fps計算
                        if count == max_count:
                            count = fps(tm)

                        #カウントを行う
                        num += 1
                        frame += 1
                        count += 1

                        #escを押したらbreakする
                        k = cv.waitKey(1)
                        if k == 27 : break # esc pressed
    else:
        #init_groupが負の時に実行
        #歩行者群検出を行う
        if not init_group:
            init_group, tracker, num, frame, count = init()
            num_groups, stats_group, center_group, center_pre_group = brack_detect_group(image, cliplimit, tilegrid, v, gauss_group, sigma_group, kernel_dilate, kernel_erode, area_min_group, area_max_group)
            tracking_group(tracker, image, num_groups, stats_group)
            init_group = True #init_groupをTrueにして検出は行わないようにする

        #もし黒色が検出されなかったらループの最初に戻る
        #init_onceをFalseにして再び黒色を検出する
        if num_groups == 0:
            print("Not detect brack color!")
            cv.imshow('OpenCV Window', image)
            init_group = False
            k = cv.waitKey(1)
            if k == 27 : 
                break # esc pressed

        #フレームが一定回数更新されたら初期化
        elif num == 10000:
            init_group, tracker, num, frame, count = init()
        else:
            #trackerの更新を行う
            ok, boxes = tracker.update(image)

            #trackerの更新ができなかったら検出からやり直し
            if not ok:
                init_group, tracker, num, frame, count = init()
            else:
                #マーカの検出を行う
                marker_cen = marker_detect(marker_cen, aruco, dictionary, image, cliplimit, tilegrid)

                #歩行者検出
                num_persons, stats_person, center_person, center_pre_person = brack_detect_person(image, v, cliplimit, tilegrid, gauss_person, sigma_person, kernel_person, area_min_person, area_max_person)

                #歩行者半径を求める
                r_person, r_person_mean = cal_r_person(num_persons, stats_person)

                #検出物体の中心座標，楕円のwidth•heightを取得
                center_group, width_height = tracking_update(boxes)

                #マーカーが検出できなかったら
                if marker_cen[0] is None or marker_cen[1] is None:
                    #数フレーム毎に歩行者の進行方向を出力
                    if frame == 2:
                    
                        #歩行者群の進行方向角度差，x軸からの楕円角度
                        theta_group, theta_ellipse, vector_e = cal_theta_group(num_groups, center_group, center_pre_group, vector_goal)

                        #描画する関数
                        drawing_vector(image, num_groups, center_group, center_pre_group)
                        all_drawing_detection_without_marker_group(image, boxes, theta_ellipse)

                        #center_preを更新する
                        center_pre_group = center_group

                        #frameを初期化する
                        frame = 0

                        #処理結果を表示
                        cv.imshow("OpenCV Window", image)

                        #fps計算
                        if count == max_count:
                            count = fps(tm)

                        #カウントを行う
                        num += 1
                        frame += 1
                        count += 1

                        #escを押したらbreakする
                        k = cv.waitKey(1)
                        if k == 27 : break # esc pressed
                    else:
                        #描画する関数
                        drawing_detection_without_marker_group(image, boxes)

                        #処理結果を表示
                        cv.imshow("OpenCV Window", image)

                        #fps計算
                        if count == max_count:
                            count = fps(tm)

                        #カウントを行う
                        num += 1
                        frame += 1
                        count += 1

                        #escを押したらbreakする
                        k = cv.waitKey(1)
                        if k == 27 : break # esc pressed
                else:
                    #ロボットがゴールに到達したらfor文を抜ける
                    if np.sqrt((marker_cen[0]*transfer(x, real_dist)-goal[0])**2+(marker_cen[1]*transfer(x, real_dist)-goal[1])**2) < 0.1:
                        break 

                    #数フレーム毎に歩行者の進行方向を出力
                    if frame == 2:
                        #LJポテンシャル計算
                        #速度計算
                        theta_group, theta_ellipse, vector_e, goryu, vx, vy =cal_velocity_group(marker_cen, num_groups, num_persons, center_group, center_pre_group, width_height, center_person, r_person, r_person_mean)

                        #合流したら速度をロボットに渡す前にwhile文の最初に戻り，歩行者追跡に切り替える
                        if goryu_jouken:
                            continue

                        #tcp通信
                        #tcp(vx,vy)

                        #ここに出力するプログラミングを書く

                        #描画する関数
                        drawing_vector(image, num_groups, center_group, center_pre_group)
                        all_drawing_detection_group(image, boxes, marker_cen, theta_ellipse)
                        drawing_goryu(image, goryu)

                        #center_preを更新する
                        center_pre_group = center_group

                        #frameを初期化する
                        frame = 0

                        #処理結果を表示
                        cv.imshow("OpenCV Window", image)

                        #fps計算
                        if count == max_count:
                            count = fps(tm)

                        #カウントを行う
                        num += 1
                        frame += 1
                        count += 1

                        #escを押したらbreakする
                        k = cv.waitKey(1)
                        if k == 27 : break # esc pressed
                    else:
                        #描画する関数
                        drawing_detection_group(image, boxes, marker_cen)

                        #処理結果を表示
                        cv.imshow("OpenCV Window", image)

                        #fps計算
                        if count == max_count:
                            count = fps(tm)

                        #カウントを行う
                        num += 1
                        frame += 1
                        count += 1

                        #escを押したらbreakする
                        k = cv.waitKey(1)
                        if k == 27 : break # esc pressed

#カメラを手放す
camera.release()
cv.destroyAllWindows()

##もし何かボタンを押せば，歩行者検出，歩行者群検出を入れ替える