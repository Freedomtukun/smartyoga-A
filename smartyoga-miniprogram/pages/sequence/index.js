const cloudSequenceService = require('../../utils/cloud-sequence-service.js');
const sequenceService = require('../../utils/sequence-service.js');
const getText = v => (typeof v === 'object' ? (v.zh || v.en || '') : v);

Page({
  data: {
    level: '',
    currentSequence: null,
    currentPoseIndex: 0,
    isPlaying: false,
    timeRemaining: 0,
    loading: true,
    error: null,
    timerId: null
  },

  onLoad(options) {
    const level = options.level || 'beginner';
    this.setData({ level });
    this.loadSequenceData(level);
  },

  async loadSequenceData(level) {
    this.setData({ loading: true, error: null });
    wx.showLoading({ title: '加载中...' });
    try {
      const data = await cloudSequenceService.getProcessedSequence(level);
      if (data && data.poses && data.poses.length > 0) {
        const state = sequenceService.setSequence(data);
        this.setData({ ...state, loading: false });
        wx.setNavigationBarTitle({
          title: `${getText(state.currentSequence.name)} - ${state.currentPoseIndex + 1}/${state.currentSequence.poses.length}`
        });
      } else {
        throw new Error('加载的序列数据无效');
      }
    } catch (err) {
      console.error('[LOAD] Failed:', err);
      this.setData({ loading: false, error: '无法加载序列数据，请稍后重试。', currentSequence: null });
      wx.setNavigationBarTitle({ title: '加载错误' });
    }
    wx.hideLoading();
  },

  startTimer() {
    if (this.data.timerId) clearInterval(this.data.timerId);
    const id = setInterval(() => {
      if (this.data.timeRemaining > 0) {
        this.setData({ timeRemaining: this.data.timeRemaining - 1 });
      } else {
        clearInterval(this.data.timerId);
        this.setData({ timerId: null });
        if (this.data.isPlaying) {
          this.handleNext();
        }
      }
    }, 1000);
    this.setData({ timerId: id });
  },

  stopTimer() {
    if (this.data.timerId) {
      clearInterval(this.data.timerId);
      this.setData({ timerId: null });
    }
  },

  playAudioGuidance(src) {
    return new Promise((resolve, reject) => {
      if (!src) return reject(new Error('No audio src'));
      const ctx = wx.createInnerAudioContext({ useWebAudioImplement: false });
      ctx.src = src;
      ctx.onEnded(() => { ctx.destroy(); resolve(); });
      ctx.onError(() => { ctx.destroy(); reject(); });
      ctx.play();
    });
  },

  togglePlayPause() {
    const { isPlaying_new } = sequenceService.togglePlayPause(this.data.isPlaying);
    this.setData({ isPlaying: isPlaying_new });
    if (isPlaying_new) {
      const pose = this.data.currentSequence.poses[this.data.currentPoseIndex];
      this.playAudioGuidance(pose.audioGuide).catch(() => {});
      this.startTimer();
    } else {
      this.stopTimer();
    }
  },

  handleNext() {
    this.stopTimer();
    const { currentSequence, currentPoseIndex } = this.data;
    const nextState = sequenceService.nextPose(currentSequence, currentPoseIndex);
    if (nextState) {
      this.setData({
        currentPoseIndex: nextState.currentPoseIndex_new,
        timeRemaining: nextState.timeRemaining_new
      });
      wx.setNavigationBarTitle({
        title: `${getText(currentSequence.name)} - ${nextState.currentPoseIndex_new + 1}/${currentSequence.poses.length}`
      });
      if (this.data.isPlaying) {
        const pose = currentSequence.poses[nextState.currentPoseIndex_new];
        this.playAudioGuidance(pose.audioGuide).catch(() => {});
        this.startTimer();
      }
    } else {
      wx.showToast({ title: '序列完成!', icon: 'success' });
      setTimeout(() => wx.redirectTo({ url: '/pages/index/index' }), 1500);
    }
  },

  handleBack() {
    this.stopTimer();
    wx.navigateBack();
  },

  onHide() {
    this.stopTimer();
  },

  onUnload() {
    this.stopTimer();
  }
});
