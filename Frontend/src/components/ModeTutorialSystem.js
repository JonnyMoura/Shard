export function createModeTutorialSystem() {
    return {
        shownModes: new Set(),
        messages: {
            save: 'Click on sounds to select them, then save them individually or as a combination to your library',
            evolve: 'Drag sounds up or down to rate them — higher means better. Switch categories using the tabs above, then submit your ratings',
            library: 'Browse your saved sounds. Click on one to see details, play it, or export it'
        },
        displayDuration: 6000,
        _timeout: null,
        _element: null,

        init() {
            if (this._element) return;

            this._element = document.createElement('div');
            this._element.className = 'mode-tutorial-message';
            document.body.appendChild(this._element);
        },

        show(modeName) {
            if (!modeName || this.shownModes.has(modeName)) return;
            if (!this.messages[modeName]) return;

            this.init();
            this.shownModes.add(modeName);

            if (this._timeout) {
                clearTimeout(this._timeout);
                this._timeout = null;
            }

            this._element.textContent = this.messages[modeName];
            this._element.classList.remove('fade-out');
            requestAnimationFrame(() => {
                this._element.classList.add('visible');
            });

            this._timeout = setTimeout(() => {
                this.hide();
            }, this.displayDuration);
        },

        hide() {
            if (!this._element) return;

            this._element.classList.add('fade-out');
            this._element.classList.remove('visible');

            if (this._timeout) {
                clearTimeout(this._timeout);
                this._timeout = null;
            }
        },

        reset() {
            this.shownModes.clear();
            this.hide();
        }
    };
}
