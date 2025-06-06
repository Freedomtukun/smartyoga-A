// 页面已精简，只保留三大入口：姿势序列、冥想、上传姿势照片检测
// 已移除首页直接拍照上传与结果展示逻辑

Page({
  handleSequencePress(event) {
    const level = event.currentTarget.dataset.level || 'beginner';
    wx.navigateTo({ url: `/pages/sequence/index?level=${level}` });
  },

  handleMeditationPress() {
    wx.navigateTo({ url: '/pages/meditation/index' });
  },

  handleUploadPhoto() {
    wx.navigateTo({ url: '/pages/photo-detect/photo-detect' });
  }
});
