import { menuIcon } from '../assets.js';

export function createNavigationStructure() {
    const hamburgerContainer = document.createElement('div');
    hamburgerContainer.className = 'hamburger-menu';

    const hamburgerButton = document.createElement('button');
    hamburgerButton.className = 'hamburger-button';
    hamburgerButton.innerHTML = `<img src="${menuIcon}" alt="Menu" />`;

    const optionsContainer = document.createElement('div');
    optionsContainer.className = 'hamburger-options';

    hamburgerContainer.appendChild(hamburgerButton);
    hamburgerContainer.appendChild(optionsContainer);
    document.body.appendChild(hamburgerContainer);

    const hoverMessage = document.createElement('div');
    hoverMessage.className = 'hamburger-hover-message';
    document.body.appendChild(hoverMessage);

    const messages = {
        evolve: 'Rate generated sounds based on your preferences to evolve new generations tailored to your taste',
        save: 'Save your favorite sounds from the current generation to your personal library',
        library: 'Browse and manage all your saved sounds and combinations',
        soundscape: 'Play or stop the ambient soundscape'
    };

    let currentMessage = '';
    let isTransitioning = false;

    const showHoverMessage = (mode) => {
        const modeManager = window.modeManager;
        const anyModeActive = modeManager?.isActive('save')
            || modeManager?.isActive('evolve')
            || modeManager?.isActive('library');

        if (anyModeActive) {
            hideHoverMessage();
            return;
        }

        if (!messages[mode]) {
            return;
        }

        const newMessage = messages[mode];
        if (currentMessage === newMessage && hoverMessage.classList.contains('visible')) {
            return;
        }

        if (currentMessage !== newMessage && hoverMessage.classList.contains('visible')) {
            if (isTransitioning) {
                return;
            }

            isTransitioning = true;
            hoverMessage.classList.remove('visible');
            setTimeout(() => {
                hoverMessage.textContent = newMessage;
                currentMessage = newMessage;
                hoverMessage.classList.add('visible');
                setTimeout(() => {
                    isTransitioning = false;
                }, 400);
            }, 400);
        } else {
            hoverMessage.textContent = newMessage;
            currentMessage = newMessage;
            hoverMessage.classList.add('visible');
        }
    };

    const hideHoverMessage = () => {
        hoverMessage.classList.remove('visible');
        setTimeout(() => {
            if (!hoverMessage.classList.contains('visible')) {
                currentMessage = '';
                isTransitioning = false;
            }
        }, 400);
    };

    window.navigationContainers = {
        hamburgerMenu: optionsContainer,
        hoverMessage: {
            show: showHoverMessage,
            hide: hideHoverMessage
        },
        topLeft: document.createElement('div'),
        topRight: document.createElement('div'),
        bottomLeft: document.createElement('div'),
        bottomRight: document.createElement('div')
    };

    let isMenuOpen = false;
    hamburgerButton.addEventListener('click', () => {
        isMenuOpen = !isMenuOpen;
        hamburgerButton.classList.toggle('open', isMenuOpen);
        optionsContainer.classList.toggle('open', isMenuOpen);

        if (!isMenuOpen) {
            hideHoverMessage();
        }
    });

    document.addEventListener('click', (event) => {
        if (!hamburgerContainer.contains(event.target) && isMenuOpen) {
            const isSaveModeActive = window.modeManager?.isActive('save');
            if (!isSaveModeActive) {
                isMenuOpen = false;
                hamburgerButton.classList.remove('open');
                optionsContainer.classList.remove('open');
                hideHoverMessage();
            }
        }
    });
}
