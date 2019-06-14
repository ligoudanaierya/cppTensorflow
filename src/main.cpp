//#include<opencv2/opencv.hpp>
//#include<opencv2/core/core.hpp>
//#include<opencv2/highgui/highgui.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
//#include<string>
//#include<iostream>
//#include<vector>
//#include"tensorflow/core/public/session.h"
//#include"tensorflow/core/platform/env.h"
#include"model_loader.h"





int main(int argc,char* argv[])
{
    std::string video_path = "../MyVideo_211.mp4";
    int a = i3dpredict(video_path);
    return 0;


}
