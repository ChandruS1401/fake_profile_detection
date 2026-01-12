// File: static/script.js

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    
    // Animate probability bars on result page
    const probabilityBars = document.querySelectorAll('.real-prob, .fake-prob');
    if (probabilityBars.length > 0) {
        setTimeout(() => {
            probabilityBars.forEach(bar => {
                const originalWidth = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = originalWidth;
                }, 100);
            });
        }, 500);
    }
    
    // Feature importance animation
    const importanceFills = document.querySelectorAll('.importance-fill');
    if (importanceFills.length > 0) {
        setTimeout(() => {
            importanceFills.forEach(fill => {
                const originalWidth = fill.style.width;
                fill.style.width = '0%';
                setTimeout(() => {
                    fill.style.width = originalWidth;
                }, 300);
            });
        }, 800);
    }
    
    // Copy data to clipboard functionality
    const copyButtons = document.querySelectorAll('.copy-btn');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const data = this.getAttribute('data-copy');
            navigator.clipboard.writeText(data).then(() => {
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                this.style.background = '#10b981';
                
                setTimeout(() => {
                    this.textContent = originalText;
                    this.style.background = '';
                }, 2000);
            });
        });
    });
    
    // Form validation for index page
    const predictionForm = document.querySelector('.prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            let isValid = true;
            const inputs = this.querySelectorAll('input[required], select[required]');
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    isValid = false;
                    input.style.borderColor = '#ef4444';
                    input.style.boxShadow = '0 0 0 3px rgba(239, 68, 68, 0.1)';
                    
                    setTimeout(() => {
                        input.style.borderColor = '';
                        input.style.boxShadow = '';
                    }, 2000);
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                alert('Please fill in all required fields.');
            }
        });
    }
    
    // Real-time calculations for index page
    const followersInput = document.getElementById('followers');
    const followingInput = document.getElementById('following');
    const ratioDisplay = document.createElement('div');
    
    if (followersInput && followingInput) {
        ratioDisplay.className = 'ratio-display';
        ratioDisplay.style.marginTop = '10px';
        ratioDisplay.style.fontSize = '0.9rem';
        ratioDisplay.style.color = '#6b7280';
        
        followingInput.parentNode.appendChild(ratioDisplay);
        
        function updateRatio() {
            const followers = parseFloat(followersInput.value) || 0;
            const following = parseFloat(followingInput.value) || 0;
            
            if (followers > 0) {
                const ratio = (following / followers).toFixed(2);
                let message = `Following/Followers Ratio: ${ratio}`;
                
                if (ratio > 3) {
                    message += ' ⚠️ (High - Suspicious)';
                    ratioDisplay.style.color = '#dc2626';
                } else if (ratio > 1.5) {
                    message += ' ℹ️ (Moderate)';
                    ratioDisplay.style.color = '#f59e0b';
                } else {
                    message += ' ✅ (Normal)';
                    ratioDisplay.style.color = '#10b981';
                }
                
                ratioDisplay.textContent = message;
            } else {
                ratioDisplay.textContent = 'Enter follower count to see ratio';
                ratioDisplay.style.color = '#6b7280';
            }
        }
        
        followersInput.addEventListener('input', updateRatio);
        followingInput.addEventListener('input', updateRatio);
        
        // Initial calculation
        updateRatio();
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId !== '#') {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 80,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });
    
    // Print functionality
    const printButton = document.createElement('button');
    printButton.textContent = '🖨️ Print Report';
    printButton.className = 'btn-secondary';
    printButton.style.marginTop = '20px';
    printButton.onclick = function() {
        window.print();
    };
    
    const actionButtons = document.querySelector('.action-buttons');
    if (actionButtons) {
        actionButtons.parentNode.insertBefore(printButton, actionButtons.nextSibling);
    }
});