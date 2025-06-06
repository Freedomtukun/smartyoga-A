import { DETECT_POSE_URL } from '../../utils/yoga-api.js';

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

  handleUploadPhoto() {
    wx.navigateTo({
      url: '/pages/photo-detect/photo-detect'
    });
  }
});
