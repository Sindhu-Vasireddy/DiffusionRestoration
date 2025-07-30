import sys
import lightning

class ProgressBar(lightning.pytorch.callbacks.TQDMProgressBar):
    """ Custom progress bar that removes the validation progress to avoid
        printing multiple lines during sampling.
    """
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True

        return bar