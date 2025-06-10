import { DETECT_POSE_URL } from '../../utils/yoga-api.js';

Page({
  data: {
    showWelcomeVideo: false,
    welcomeVideoUrl: ''
  },

  onLoad() {
    const hasWatched = wx.getStorageSync('hasWatchedWelcomeVideo');
    if (!hasWatched) {
      this.setData({
        showWelcomeVideo: true,
        welcomeVideoUrl: 'https://yogasmart-static-1351554677.cos.ap-shanghai.myqcloud.com/assets/welcome.mp4'
      });
    }
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
    wx.navigateTo({
      url: '/pages/photo-detect/photo-detect'
    });
  },

  onWelcomeVideoEnd() {
    this.hideWelcomeVideo();
  },

  skipWelcomeVideo() {
    this.hideWelcomeVideo();
  },

  hideWelcomeVideo() {
    this.setData({ showWelcomeVideo: false });
    wx.setStorageSync('hasWatchedWelcomeVideo', true);
  }
});
