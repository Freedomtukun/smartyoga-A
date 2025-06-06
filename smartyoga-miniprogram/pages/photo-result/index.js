Page({
  data: {
    score: 0,
    skeletonUrl: '',
    suggestion: ''
  },

  onLoad(options) {
    this.setData({
      score: options.score || 0,
      skeletonUrl: options.skeletonUrl ? decodeURIComponent(options.skeletonUrl) : '',
      suggestion: options.suggestion ? decodeURIComponent(options.suggestion) : ''
    });
  },

  onShareAppMessage() {
    return {
      title: `我的瑜伽得分 ${this.data.score}`,
      path: '/pages/index/index'
    };
  }
});
