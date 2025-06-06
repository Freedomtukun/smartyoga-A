// app.js
App({
  onLaunch: function () {
    if (!wx.cloud) {
      console.error('Please update your WeChat version to use cloud capabilities.');
    } else {
      wx.cloud.init({
        // env: wx.cloud.DYNAMIC_CURRENT_ENV, // Dynamically use current environment
        env: 'smartyoga-3gh3muyt510ddd2d', // Specify your cloud environment ID
        traceUser: true, // Recommended for user tracking
      });
    }
    console.log('Cloud Env Initialized with ID: smartyoga-3gh3muyt510ddd2d');
  },
  globalData: {
    // You can add global data here
  }
});
