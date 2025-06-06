/**
 * @typedef {Object} Pose
 * @property {string} id
 * @property {number} duration
 * @property {number} [breathCount]
 * @property {string} [audioGuide] // This is a filename, e.g., "pose_1_mountain_tadasa.mp3"
 * @property {{en: string, zh: string}} instructions
 * @property {{en: string, zh: string}} [transitionHint]
 */

/**
 * @typedef {Object} Sequence
 * @property {string} id
 * @property {{en: string, zh: string}} name
 * @property {number} difficulty // e.g., 1 for beginner, 2 for intermediate
 * @property {number} duration // Total sequence duration in seconds
 * @property {{en: string, zh: string}} description
 * @property {Pose[]} poses
 * @property {{introduction: string, backgroundMusic: string}} audioGuide // Filenames/paths
 */

/**
 * Initializes or updates the sequence state.
 * @param {Sequence | null} newSequence - The new sequence to set. Can be null to clear.
 * @param {number} [initialPoseIndex=0] - The initial pose index to start from.
 * @returns {object} An object with the new state values for currentSequence, currentPoseIndex, isPlaying, and timeRemaining.
 */
function setSequence(newSequence, initialPoseIndex = 0) {
  if (!newSequence || !newSequence.poses || newSequence.poses.length === 0) {
    return {
      currentSequence: null,
      currentPoseIndex: 0,
      isPlaying: false,
      timeRemaining: 0,
    };
  }
  // Ensure initialPoseIndex is valid
  const poseIndex = Math.max(0, Math.min(initialPoseIndex, newSequence.poses.length - 1));
  
  return {
    currentSequence: newSequence,
    currentPoseIndex: poseIndex,
    isPlaying: false, // Typically starts paused
    timeRemaining: newSequence.poses[poseIndex]?.duration || 0,
  };
}

/**
 * Calculates the state for the next pose in the sequence.
 * @param {Sequence} currentSequence - The current sequence object from page data.
 * @param {number} currentPoseIndex - The current pose index from page data.
 * @returns {object | null} An object with { currentPoseIndex_new, timeRemaining_new, isPlaying_new } or null if at the end.
 */
function nextPose(currentSequence, currentPoseIndex) {
  if (currentSequence && currentSequence.poses && currentPoseIndex < currentSequence.poses.length - 1) {
    const nextIndex = currentPoseIndex + 1;
    return {
      currentPoseIndex_new: nextIndex,
      timeRemaining_new: currentSequence.poses[nextIndex].duration,
      // isPlaying_new: true, // Let the page decide if it should auto-play.
                           // This utility should focus on state transition values.
    };
  }
  return null; // Indicates sequence end or invalid state
}

/**
 * Calculates the state for the previous pose in the sequence.
 * @param {Sequence} currentSequence - The current sequence object from page data.
 * @param {number} currentPoseIndex - The current pose index from page data.
 * @returns {object | null} An object with { currentPoseIndex_new, timeRemaining_new, isPlaying_new } or null if at the beginning.
 */
function previousPose(currentSequence, currentPoseIndex) {
  if (currentSequence && currentSequence.poses && currentPoseIndex > 0) {
    const prevIndex = currentPoseIndex - 1;
    return {
      currentPoseIndex_new: prevIndex,
      timeRemaining_new: currentSequence.poses[prevIndex].duration,
      // isPlaying_new: false, // Let the page decide this.
    };
  }
  return null; // Indicates at the start or invalid state
}

/**
 * Calculates the new playing state.
 * @param {boolean} isPlaying - The current playing state.
 * @returns {object} An object with { isPlaying_new: new_state }.
 */
function togglePlayPause(isPlaying) {
  return { isPlaying_new: !isPlaying };
}

/**
 * Returns the new time remaining state.
 * @param {number} time - The new time remaining in seconds.
 * @returns {object} An object with { timeRemaining_new: time }.
 */
function setTimeRemaining(time) {
  // Ensure time is not negative
  return { timeRemaining_new: Math.max(0, time) };
}

/**
 * Returns an object representing the initial state for sequence-related data.
 * @returns {object} An object with currentSequence, currentPoseIndex, isPlaying, and timeRemaining set to initial values.
 */
function resetSequence() {
  return {
    currentSequence: null,
    currentPoseIndex: 0,
    isPlaying: false,
    timeRemaining: 0,
  };
}

module.exports = {
  setSequence,
  nextPose,
  previousPose,
  togglePlayPause,
  setTimeRemaining,
  resetSequence,
};
