Page({
  data: {
    isPlaying: false,
    audioUrl: 'https://yogasmart-static-1351554677.cos.ap-shanghai.myqcloud.com/static/audio/meditation_gentle.mp3',
  },

  onLoad: function () {
    this.innerAudioContext = wx.createInnerAudioContext({
      useWebAudioImplement: false // Use system audio player
    });
    this.innerAudioContext.src = this.data.audioUrl;
    this.innerAudioContext.loop = true;

    this.innerAudioContext.onPlay(() => {
      console.log('Audio started playing');
      this.setData({ isPlaying: true });
    });

    this.innerAudioContext.onPause(() => {
      console.log('Audio paused');
      this.setData({ isPlaying: false });
    });

    this.innerAudioContext.onStop(() => {
      console.log('Audio stopped');
      this.setData({ isPlaying: false });
    });
    
    this.innerAudioContext.onEnded(() => {
      console.log('Audio ended');
      // Due to loop=true, onEnded might not be triggered as expected in some cases.
      // We manage isPlaying state primarily via play/pause actions.
      this.setData({ isPlaying: false }); 
    });

    this.innerAudioContext.onError((res) => {
      console.error('Audio error:', res.errMsg, 'Error code:', res.errCode);
      wx.showToast({
        title: '音频播放失败',
        icon: 'none'
      });
      this.setData({ isPlaying: false });
    });
  },

  handleBack: function () {
    if (this.innerAudioContext) {
      this.innerAudioContext.stop(); // Stop releases resources, destroy might not be needed immediately unless page is unloaded
    }
    wx.navigateBack();
  },

  toggleMeditation: function () {
    if (this.data.isPlaying) {
      this.innerAudioContext.pause();
    } else {
      this.innerAudioContext.play();
    }
  },

  onUnload: function () {
    if (this.innerAudioContext) {
      this.innerAudioContext.destroy();
      console.log('Audio context destroyed');
    }
  },
});
