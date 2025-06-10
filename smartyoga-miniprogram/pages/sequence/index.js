import { uploadFrameForScoring, DEFAULT_POSE_IMAGE } from '../../utils/yoga-api.js';
import poseImages from '../../assets/images.js';
const cloudSequenceService = require('../../utils/cloud-sequence-service.js');
const sequenceService = require('../../utils/sequence-service.js');
const getText = v => (typeof v === 'object' ? (v.zh || v.en || '') : v);

function normalizeKey(key) {
  if (!key) return '';
  return key.toLowerCase().replace(/[-\s]/g, '_');
}

Page({
  data: {
    level: '',
    currentSequence: null,
    currentPoseIndex: 0,
    isPlaying: false,
    timeRemaining: 0,
    loading: true,
    error: null,
    skeletonUrl: null,
    timerId: null,
    showScoreModal: false,
    poseScore: null,
    scoreSkeletonImageUrl: null,
    defaultPoseImage: DEFAULT_POSE_IMAGE,
    poseImages,
    normalizedPoseKey: ''
  },

  updateNormalizedPoseKey() {
    const seq = this.data.currentSequence;
    const idx = this.data.currentPoseIndex;
    const key = seq && seq.poses && seq.poses[idx] ? seq.poses[idx].key : '';
    this.setData({ normalizedPoseKey: normalizeKey(key) });
  },



  // Page lifecycle: Load sequence data
  onLoad: function (options) {
    const level = options.level || 'beginner';
    this.setData({ level: level });
    this.loadSequenceData(level);
  },

  // Load sequence data
  async loadSequenceData(level) {
    console.log('[LOAD] Loading sequence for level:', level);
    this.setData({ loading: true, error: null });
    wx.showLoading({ title: '加载中...' });
    
    try {
      const sequenceData = await cloudSequenceService.getProcessedSequence(level);

      if (sequenceData && sequenceData.poses && sequenceData.poses.length > 0) {
        // Auto fill pose key using image_url filename if missing
        sequenceData.poses = sequenceData.poses.map(pose => {
          if (!pose.key && pose.image_url) {
            const file = pose.image_url.split('/').pop();
            pose.key = file.replace(/\.(png|jpg|jpeg)$/i, '');
          }
          return pose;
        });
        const initialState = sequenceService.setSequence(sequenceData);
        this.setData({
          ...initialState,
          loading: false
        });
        this.updateNormalizedPoseKey();
        wx.hideLoading();
        wx.setNavigationBarTitle({ 
          title: `${getText(initialState.currentSequence.name)} - ${initialState.currentPoseIndex + 1}/${initialState.currentSequence.poses.length}` 
        });
      } else {
        console.error('[LOAD] Invalid sequence data:', sequenceData);
        throw new Error('加载的序列数据无效');
      }
    } catch (err) {
      console.error('[LOAD] Failed to load sequence:', err);
      let userErrorMessage = '无法加载序列数据，请稍后重试。';
      let toastMessage = '加载失败，请稍后重试';

      if (err && err.message === 'MISSING_SIGNED_URL') {
        userErrorMessage = '序列配置获取失败，请检查网络或稍后重试。';
        toastMessage = '序列配置获取失败';
      }
      
      this.setData({ 
        loading: false, 
        error: userErrorMessage, 
        currentSequence: null 
      });
      wx.hideLoading();
      wx.showToast({ title: toastMessage, icon: 'none' });
      wx.setNavigationBarTitle({ title: '加载错误' });
    }
  },

  // Timer management
  startTimer: function () {
    if (this.data.timerId) clearInterval(this.data.timerId);

    const timerId = setInterval(() => {
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
    this.setData({ timerId: timerId });
  },

  stopTimer: function () {
    if (this.data.timerId) {
      clearInterval(this.data.timerId);
      this.setData({ timerId: null });
    }
  },

  // Play audio guidance
  playAudioGuidance: function (src) {
    return new Promise((resolve, reject) => {
      if (!src) {
        console.warn('[AUDIO] No audio src provided');
        reject(new Error("No audio src provided"));
        return;
      }

      const audioCtx = wx.createInnerAudioContext({ useWebAudioImplement: false });
      audioCtx.src = src;
      audioCtx.onEnded(() => { 
        audioCtx.destroy(); 
        resolve(); 
      });
      audioCtx.onError((error) => {
        console.error('[AUDIO] Error playing:', src, error);
        wx.showToast({ title: '音频播放失败', icon: 'none' });
        audioCtx.destroy();
        reject(error);
      });
      audioCtx.play();
    });
  },

  // Navigation handlers
  handleBack: function () {
    this.stopTimer();
    wx.navigateBack();
  },

  handleNext: function () {
    this.stopTimer();
    const { currentSequence, currentPoseIndex } = this.data;
    const nextState = sequenceService.nextPose(currentSequence, currentPoseIndex);

    if (nextState) {
      this.setData({
        currentPoseIndex: nextState.currentPoseIndex_new,
        timeRemaining: nextState.timeRemaining_new
      });
      this.updateNormalizedPoseKey();
      wx.setNavigationBarTitle({
        title: `${getText(currentSequence.name)} - ${nextState.currentPoseIndex_new + 1}/${currentSequence.poses.length}`
      });
      
      if (this.data.isPlaying) {
        const newCurrentPose = currentSequence.poses[nextState.currentPoseIndex_new];
        this.playAudioGuidance(newCurrentPose.audioGuide)
          .catch(e => console.error("[AUDIO] Error in handleNext:", e));
        this.startTimer();
      }
    } else {
      const randomScore = Math.floor(Math.random() * 30) + 70;
      this.setData({
        poseScore: { score: randomScore, feedback: '保持呼吸流畅，继续练习' },
        showScoreModal: true
      });
    }
  },

  closeScoreModal() {
    this.setData({ showScoreModal: false });
    wx.redirectTo({ url: '/pages/index/index' });
  },

  // Toggle play/pause
  togglePlayPause: function () {
    const { isPlaying_new } = sequenceService.togglePlayPause(this.data.isPlaying);
    this.setData({ isPlaying: isPlaying_new });

    if (isPlaying_new) {
      const currentPose = this.data.currentSequence.poses[this.data.currentPoseIndex];
      this.playAudioGuidance(currentPose.audioGuide)
        .catch(e => console.error("[AUDIO] Error in togglePlayPause:", e));
      this.startTimer();
    } else {
      this.stopTimer();
    }
  },



  // Image error handler - use placeholder
  onImageError: function(e) {
    const dataset = e.currentTarget.dataset;
    const imageType = dataset.type;
    const imageIndex = dataset.index;
    
    console.warn('[IMAGE_ERROR] Failed to load image:', e.detail.errMsg, 'Type:', imageType, 'Index:', imageIndex);
    
    // Update the specific image that failed based on type
    if (imageType === 'skeleton' && imageIndex !== undefined) {
      // Update skeleton image in topThreeFrames
      const frameIndex = parseInt(imageIndex);
      if (!isNaN(frameIndex) && this.data.topThreeFrames[frameIndex]) {
        this.setData({
          [`topThreeFrames[${frameIndex}].skeletonUrl`]: DEFAULT_POSE_IMAGE
        });
      }
    }
  },

  // Lifecycle hooks
  onShow: function () {
    wx.setKeepScreenOn({ keepScreenOn: true });
  },
  onHide: function () {
    wx.setKeepScreenOn({ keepScreenOn: false });
    this.stopTimer();
  },

  onUnload: function () {
    this.stopTimer();
    // Cancel any ongoing uploads
    if (this.data.currentUploadTasks && this.data.currentUploadTasks.length > 0) {
      console.log('[UNLOAD] Cancelling', this.data.currentUploadTasks.length, 'ongoing uploads');
      this.data.currentUploadTasks.forEach(task => {
        if (task && typeof task.abort === 'function') {
          task.abort();
        }
      });
    }
  }
});
