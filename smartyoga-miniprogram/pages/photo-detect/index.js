import { DETECT_POSE_URL } from '../../utils/yoga-api.js';

Page({
  data: {
    poseId: 'mountain_pose'
  },

  choosePhoto() {
    const poseId = this.data.poseId;
    wx.chooseImage({
      count: 1,
      sizeType: ['original', 'compressed'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        const tempFile = res.tempFilePaths[0];
        wx.showLoading({ title: '上传中...', mask: true });
        wx.uploadFile({
          url: DETECT_POSE_URL,
          filePath: tempFile,
          name: 'file',
          formData: { poseId },
          success: (uploadRes) => {
            wx.hideLoading();
            let data = {};
            try {
              data = JSON.parse(uploadRes.data);
            } catch (e) {}
            if (data.code === 'SUCCESS' || data.code === 'OK') {
              const url = `/pages/photo-result/index?score=${data.score}&skeletonUrl=${encodeURIComponent(data.skeletonUrl || '')}&suggestion=${encodeURIComponent(data.suggestion || data.feedback || '')}`;
              wx.navigateTo({ url });
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
