import { DETECT_POSE_URL } from '../../utils/yoga-api.js';

Page({
  data: {
    poseId: 'mountain_pose',
    photoResult: null
  },

  handleSequencePress(event) {
    const level = event.currentTarget.dataset.level;
    wx.navigateTo({
      url: `/pages/sequence/index?level=${level}`,
    });
  },

  handleMeditationPress() {
    wx.navigateTo({
      url: '/pages/meditation/index',
    });
  },

  handleUploadPhoto() {
    const poseId = this.data.poseId;
    wx.chooseImage({
      count: 1,
      sizeType: ['original', 'compressed'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        const tempFilePath = res.tempFilePaths[0];
        wx.showLoading({ title: '上传中...', mask: true });
        wx.uploadFile({
          url: DETECT_POSE_URL,
          filePath: tempFilePath,
          name: 'file',
          formData: { poseId },
          success: (uploadRes) => {
            wx.hideLoading();
            let data = {};
            try {
              data = JSON.parse(uploadRes.data);
            } catch (e) {
              wx.showToast({ title: '解析失败', icon: 'none' });
              return;
            }

            if (data.code === 'SUCCESS' || data.code === 'OK') {
              this.setData({
                photoResult: {
                  score: data.score,
                  skeletonUrl: data.skeletonUrl,
                  suggestion: data.suggestion || data.feedback || ''
                }
              });
            } else {
              wx.showToast({ title: data.msg || '识别失败', icon: 'none' });
            }
          },
          fail: () => {
            wx.hideLoading();
            wx.showToast({ title: '上传失败', icon: 'none' });
          }
        });
      }
    });
  }
});
