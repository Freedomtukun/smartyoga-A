// 导入冥想数据
const meditationSessions = require('../../assets/meditation_sessions.js');
Page({
  data: {
    // 音频播放状态
    isPlaying: false,
    // 冥想会话列表
    meditationList: [],
    // 当前选中的冥想项索引
    currentIndex: 0,
    // 当前选中的冥想项详情
    currentMeditation: {}
  },

  /**
   * 页面加载时初始化
   */
  onLoad: function () {
    this.initMeditationData();
    this.initAudioContext();
  },

  /**
   * 初始化冥想数据
   */
  initMeditationData: function () {
    const meditationList = meditationSessions || [];
    const currentMeditation = meditationList[0] || {};
    
    this.setData({
      meditationList: meditationList,
      currentMeditation: currentMeditation,
      currentIndex: 0
    });
    
    console.log('冥想数据加载完成:', meditationList);
  },

  /**
   * 初始化音频上下文
   */
  initAudioContext: function () {
    // 销毁之前的音频上下文
    if (this.innerAudioContext) {
      this.innerAudioContext.destroy();
    }

    this.innerAudioContext = wx.createInnerAudioContext({
      useWebAudioImplement: false
    });

    // 设置初始音频源
    if (this.data.currentMeditation.audioUrl) {
      this.innerAudioContext.src = this.data.currentMeditation.audioUrl;
    }

    // 绑定音频事件
    this.bindAudioEvents();
  },

  /**
   * 绑定音频事件监听
   */
  bindAudioEvents: function () {
    // 播放开始
    this.innerAudioContext.onPlay(() => {
      console.log('音频开始播放');
      this.setData({ isPlaying: true });
    });

    // 播放暂停
    this.innerAudioContext.onPause(() => {
      console.log('音频暂停');
      this.setData({ isPlaying: false });
    });

    // 播放停止
    this.innerAudioContext.onStop(() => {
      console.log('音频停止');
      this.setData({ isPlaying: false });
    });

    // 播放结束
    this.innerAudioContext.onEnded(() => {
      console.log('音频播放结束');
      this.setData({ isPlaying: false });
    });

    // 播放错误
    this.innerAudioContext.onError((res) => {
      console.error('音频播放错误:', res.errMsg, '错误代码:', res.errCode);
      wx.showToast({
        title: '音频播放失败',
        icon: 'none',
        duration: 2000
      });
      this.setData({ isPlaying: false });
    });

    // 音频加载完成
    this.innerAudioContext.onCanplay(() => {
      console.log('音频加载完成，可以播放');
    });
  },

  /**
   * 切换冥想类型
   * @param {Event} e - 事件对象
   */
  switchMeditation: function (e) {
    const index = e.currentTarget.dataset.index;
    const newMeditation = this.data.meditationList[index];
    
    if (!newMeditation) {
      console.error('未找到对应的冥想项:', index);
      return;
    }

    // 如果是同一个冥想项，直接返回
    if (index === this.data.currentIndex) {
      return;
    }

    console.log('切换冥想类型:', newMeditation.name);

    // 停止当前播放的音频
    if (this.innerAudioContext && this.data.isPlaying) {
      this.innerAudioContext.stop();
    }

    // 更新数据
    this.setData({
      currentIndex: index,
      currentMeditation: newMeditation,
      isPlaying: false
    });

    // 切换音频源
    if (this.innerAudioContext) {
      this.innerAudioContext.src = newMeditation.audioUrl;
    }

    // 显示切换提示
    wx.showToast({
      title: `已切换到${newMeditation.name}`,
      icon: 'none',
      duration: 1500
    });
  },

  /**
   * 播放/暂停冥想音频
   */
  toggleMeditation: function () {
    if (!this.innerAudioContext) {
      console.error('音频上下文未初始化');
      return;
    }

    if (!this.data.currentMeditation.audioUrl) {
      wx.showToast({
        title: '音频链接无效',
        icon: 'none'
      });
      return;
    }

    try {
      if (this.data.isPlaying) {
        // 暂停播放
        this.innerAudioContext.pause();
        console.log('暂停冥想音频');
      } else {
        // 开始播放
        this.innerAudioContext.play();
        console.log('开始播放冥想音频');
      }
    } catch (error) {
      console.error('音频播放控制出错:', error);
      wx.showToast({
        title: '音频控制失败',
        icon: 'none'
      });
    }
  },

  /**
   * 返回上一页
   */
  handleBack: function () {
    // 停止音频播放
    if (this.innerAudioContext) {
      this.innerAudioContext.stop();
    }
    wx.navigateBack();
  },

  /**
   * 格式化时长显示
   * @param {number} duration - 时长（秒）
   * @returns {string} 格式化的时长字符串
   */
  formatDuration: function (duration) {
    const minutes = Math.floor(duration / 60);
    const seconds = duration % 60;
    return `${minutes}分${seconds}秒`;
  },

  /**
   * 页面卸载时清理资源
   */
  onUnload: function () {
    if (this.innerAudioContext) {
      this.innerAudioContext.destroy();
      console.log('音频上下文已销毁');
    }
  },

  /**
   * 页面隐藏时暂停音频
   */
  onHide: function () {
    if (this.innerAudioContext && this.data.isPlaying) {
      this.innerAudioContext.pause();
    }
  }
});