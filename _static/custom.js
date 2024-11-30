// File: custom.js

/**
 * This function will make sure all links in the document open as separate tabs
 * instead of opening on the current user window.
 */
window.onload = function () {
    let links = document.getElementsByTagName('a');
    for (let l of links) {
        if (l.classList.contains("reference") && l.classList.contains("external")) {
            l.target = '_blank'; // Open links in a new tab
        }
    }
};
