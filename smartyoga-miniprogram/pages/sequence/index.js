// 序列页面简化为只展示体式并播放音频，移除录制和上传逻辑
import beginner from '../../beginner.json'
import intermediate from '../../intermediate.json'
import advanced from '../../advanced.json'

Page({
  data: {
    level: 'beginner',
    sequence: null,
    index: 0,
    isPlaying: false,
    timeRemaining: 0,
    timerId: null,
    audioCtx: null
  },

  onLoad(options) {
    const level = options.level || 'beginner'
    this.setData({ level })
    this.loadSequence(level)
  },

  loadSequence(level) {
    let sequence = beginner
    if (level === 'intermediate') sequence = intermediate
    if (level === 'advanced') sequence = advanced
    this.setData({
      sequence,
      index: 0,
      isPlaying: false,
      timeRemaining: sequence.poses[0].duration
    })
    wx.setNavigationBarTitle({
      title: `${sequence.name.zh} - 1/${sequence.poses.length}`
    })
  },

  startTimer() {
    this.stopTimer()
    this.data.timerId = setInterval(() => {
      if (this.data.timeRemaining > 0) {
        this.setData({ timeRemaining: this.data.timeRemaining - 1 })
      } else {
        this.handleNext()
      }
    }, 1000)
  },

  stopTimer() {
    if (this.data.timerId) {
      clearInterval(this.data.timerId)
      this.data.timerId = null
    }
  },

  playAudio(src) {
    if (!src) return
    if (!this.data.audioCtx) {
      this.data.audioCtx = wx.createInnerAudioContext({ useWebAudioImplement: false })
    }
    const ctx = this.data.audioCtx
    ctx.src = src
    ctx.play()
  },

  togglePlayPause() {
    if (this.data.isPlaying) {
      this.stopTimer()
      if (this.data.audioCtx) this.data.audioCtx.pause()
      this.setData({ isPlaying: false })
    } else {
      const pose = this.data.sequence.poses[this.data.index]
      this.playAudio(pose.audioGuide)
      this.startTimer()
      this.setData({ isPlaying: true })
    }
  },

  handleNext() {
    this.stopTimer()
    if (this.data.index < this.data.sequence.poses.length - 1) {
      const nextIndex = this.data.index + 1
      const pose = this.data.sequence.poses[nextIndex]
      this.setData({
        index: nextIndex,
        timeRemaining: pose.duration,
        isPlaying: false
      })
      wx.setNavigationBarTitle({
        title: `${this.data.sequence.name.zh} - ${nextIndex + 1}/${this.data.sequence.poses.length}`
      })
    } else {
      wx.showToast({ title: '序列完成', icon: 'success' })
      wx.navigateBack({ delta: 1 })
    }
  },

  handleBack() {
    this.stopTimer()
    if (this.data.audioCtx) this.data.audioCtx.stop()
    wx.navigateBack()
  },

  onUnload() {
    this.stopTimer()
    if (this.data.audioCtx) this.data.audioCtx.destroy()
  }
})
