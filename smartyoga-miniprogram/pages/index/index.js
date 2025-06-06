Page({
  data: {},

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

  handlePhotoDetect() {
    wx.navigateTo({
      url: '/pages/photo-detect/index'
    });
  }
});
