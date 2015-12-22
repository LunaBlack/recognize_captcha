# recognize_captcha
recognize captcha

文件说明：
pic为待识别的验证码，共50个，需要预处理。
model是供训练的模板字符。

初步思路：
先将背景色统一为黑色，然后灰度化原图，使之变为黑底白字。
然后，利用腐蚀/开运算去掉干扰线和噪点。
但是，效果一般。

修正思路：
在初步思路的基础上，统计原图的颜色占比。
然后，与初步思路中得到的图比对，统计各颜色在初步处理后的占比，选择前四个颜色作为字符颜色。
根据颜色，将图处理为黑底白字，作为提取字符的结果。（字符被干扰线遮挡的部分将变为黑色）

根据观察，将验证码切分为四个字符，然后进行识别分类。
采用了KNN和SVM两种算法。

该程序总体比较简单，仅能处理简单的验证码。其余情况需要进一步分析。


