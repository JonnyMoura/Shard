export const keysPressed = new Set();

const movementKeys = new Set([
    'w',
    'a',
    's',
    'd',
    'arrowup',
    'arrowdown',
    'arrowleft',
    'arrowright',
    ' ',
    'shift'
]);

export function installKeyboardControls() {
    document.addEventListener('keydown', (event) => {
        const activeElement = document.activeElement;
        const isInInput = activeElement && (
            activeElement.tagName === 'INPUT'
            || activeElement.tagName === 'TEXTAREA'
            || activeElement.isContentEditable
            || activeElement.closest('input')
            || activeElement.closest('textarea')
        );

        if (isInInput) {
            console.log('⌨️ Allowing input field:', activeElement.tagName);
            return;
        }

        if (window.__modeTransitioning) {
            console.log('⌨️ BLOCKING ALL KEYS during transition:', event.key);
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            return;
        }

        const isSaveMode = window.modeManager?.isActive('save');
        const isEvolveMode = window.modeManager?.isActive('evolve');
        const isLibraryMode = window.modeManager?.isActive('library');
        const isAnyMode = isSaveMode || isEvolveMode || isLibraryMode;

        if (isAnyMode) {
            if (event.key === 'Escape') {
                console.log('⌨️ Escape pressed - exiting mode');
                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();

                if (isSaveMode) {
                    window.modeManager?.goTo('save');
                } else if (isEvolveMode) {
                    window.modeManager?.goTo('evolve');
                } else if (isLibraryMode) {
                    window.modeManager?.goTo('library');
                }
                return;
            }

            console.log('⌨️ BLOCKING key in mode:', event.key);
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            return;
        }

        const key = event.key.toLowerCase();
        if (movementKeys.has(key)) {
            keysPressed.add(key);
            event.preventDefault();
            return;
        }

        console.log('⌨️ BLOCKING key in main scene:', event.key);
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
    }, true);

    document.addEventListener('keyup', (event) => {
        const activeElement = document.activeElement;
        const isInInput = activeElement && (
            activeElement.tagName === 'INPUT'
            || activeElement.tagName === 'TEXTAREA'
            || activeElement.isContentEditable
        );

        if (isInInput) return;

        if (window.__modeTransitioning) {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            return;
        }

        const isSaveMode = window.modeManager?.isActive('save');
        const isEvolveMode = window.modeManager?.isActive('evolve');
        const isLibraryMode = window.modeManager?.isActive('library');
        const isAnyMode = isSaveMode || isEvolveMode || isLibraryMode;

        if (isAnyMode) {
            if (event.key === 'Escape') {
                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();
                return;
            }

            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            return;
        }

        const key = event.key.toLowerCase();
        if (movementKeys.has(key)) {
            keysPressed.delete(key);
            return;
        }

        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
    }, true);

    document.addEventListener('keypress', (event) => {
        const activeElement = document.activeElement;
        const isInInput = activeElement && (
            activeElement.tagName === 'INPUT'
            || activeElement.tagName === 'TEXTAREA'
            || activeElement.isContentEditable
        );

        if (isInInput) return;

        if (window.__modeTransitioning) {
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            return;
        }

        const isSaveMode = window.modeManager?.isActive('save');
        const isEvolveMode = window.modeManager?.isActive('evolve');
        const isLibraryMode = window.modeManager?.isActive('library');
        const isAnyMode = isSaveMode || isEvolveMode || isLibraryMode;

        if (isAnyMode) {
            if (event.key === 'Escape') {
                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();
                return;
            }

            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            return;
        }

        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
    }, true);
}
