#include "stdio.h"
#include <iostream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

class LightDescriptor
{	
public:float width,length,angle,area;
   	  cv::Point2f center;
public:
    LightDescriptor() {};
    LightDescriptor(const cv::RotatedRect& light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
    const LightDescriptor& operator =(const LightDescriptor& ld)
    {
        this->width = ld.width;
        this->length = ld.length;
        this->center = ld.center;
        this->angle = ld.angle;
        this->area = ld.area;
        return *this;
    }   
};

//关键在于筛选灯条这个功能，所以封装成一个函数进行处理，轮廓、
void filterContours(vector<vector<Point> >& contours, vector<LightDescriptor>& lightInfos,Mat frame) {
    Mat c = frame.clone();
    for (int i = 0; i < contours.size(); i++) {
        // 求轮廓面积
        double area = contourArea(contours[i]);
        // 去除较小轮廓&fitEllipse的限制条件
        if (area < 10 || contours[i].size() <= 5)
            continue;//相当于就是把这段轮廓去除掉
        // 用椭圆拟合区域得到外接矩形（特殊的处理方式：因为灯条是椭圆型的，所以用椭圆去拟合轮廓，再直接获取旋转外接矩形即可）
        RotatedRect Light_Rec = fitEllipse(contours[i]);
        
        // 长宽比和轮廓面积比限制（由于要考虑灯条的远近都被识别到，所以只需要看比例即可）
        if (Light_Rec.size.width / Light_Rec.size.height > 4)
            continue;
        // 扩大灯柱的面积（相当于对灯柱做膨胀操作）
        Light_Rec.size.height *= 1.2;
        Light_Rec.size.width *= 1.2;
        
        lightInfos.push_back(LightDescriptor(Light_Rec));
    }
 
}

int main()
{
    VideoCapture video; //VC类对象化
    video.open("D:\\code\\opencv\\final_test\\blueVideo5.mp4");
    Mat frame,channels[3],binary,Gaussian,dilatee;
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Rect boundRect; 
    RotatedRect box;
    vector<vector<Point>> contours; 
    vector<Vec4i> hierarchy;    
    vector<Point2f> boxPts(4);

    for (;;) {
        Rect point_array[20];
        video >> frame;  //读取每帧
        if (frame.empty()) {
            break;
        }
        split(frame, channels); //通道分离
        threshold(channels[0], binary, 220, 255, 0);//二值化
        GaussianBlur(binary, Gaussian, Size(5, 5), 0);//滤波
        dilate(Gaussian, dilatee, element);//膨胀，把滤波得到的细灯条变宽
        findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//轮廓检测
        
        //筛选灯条，其中的lightInfos是返回的被筛选好的灯条
        vector<LightDescriptor> lightInfos;
        filterContours(contours, lightInfos,frame);
        
        //遍历所有灯条进行匹配（双重循环，在原先的i灯条的基础上去匹配剩下的灯条，即筛选出来）
        for (size_t i = 0; i < lightInfos.size(); i++) {                      
            for (size_t j = i + 1; (j < lightInfos.size()); j++) {
                LightDescriptor& leftLight = lightInfos[i];
                LightDescriptor& rightLight = lightInfos[j];
 
                //角差
                float angleDiff_ = abs(leftLight.angle - rightLight.angle);
                //长度差比率（由于远近的关系，需要考虑的应该是个差距的比值而不是差距本身）
                float LenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
                //筛选
                if (angleDiff_ > 10 || LenDiff_ratio > 0.8) {
                    continue;
                }
 
                //左右灯条相距距离
                float dis = pow(pow((leftLight.center.x - rightLight.center.x), 2) + pow((leftLight.center.y - rightLight.center.y), 2), 0.5);
                //左右灯条长度的平均值
                float meanLen = (leftLight.length + rightLight.length) / 2;
                //左右灯条长度差比值
                float lendiff = abs(leftLight.length - rightLight.length) / meanLen ;
                //左右灯条中心点y的差值
                float yDiff = abs(leftLight.center.y - rightLight.center.y);
                //y差比率
                float yDiff_ratio = yDiff / meanLen;
                //左右灯条中心点x的差值
                float xDiff = abs(leftLight.center.x - rightLight.center.x);
                //x差比率
                float xDiff_ratio = xDiff / meanLen;
                //相距距离与灯条长度比值
                float ratio = dis / meanLen;
                //筛选
                if (lendiff > 0.5 ||                   
                    yDiff_ratio > 1.2 ||
                    xDiff_ratio > 2 ||
                    xDiff_ratio < 0.6 ||
                    ratio > 3.5 ||
                    ratio < 0.5) {
                    continue;
                }

 //绘制矩形
                Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y)/2);
                RotatedRect rect = RotatedRect(center, Size(dis, meanLen), (leftLight.angle + rightLight.angle) / 2);
                Point2f vertices[4];
                rect.points(vertices);
                for (int i = 0; i < 4; i++) {
                    line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
                }                              
           }
        }
 
        namedWindow("video", WINDOW_FREERATIO);
        imshow("video",frame);
        waitKey(20);
    }
    video.release();
    cv::destroyAllWindows();
    return 0;
}