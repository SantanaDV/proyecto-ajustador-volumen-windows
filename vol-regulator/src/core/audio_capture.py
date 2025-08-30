import logging
from typing import Optional

import numpy as np
import soundcard as sc

logger = logging.getLogger(__name__)


class AudioCapture:
    def __init__(
        self,
        device_name: Optional[str],
        blocksize: int,
        sample_rate: int = 48000,
        channels: int = 2,
        exclusive: bool = False,
    ):
        # Param validation
        if not isinstance(blocksize, int) or blocksize <= 0:
            raise ValueError("blocksize must be a positive integer")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")

        self.device_name = device_name
        self.blocksize = blocksize
        self.fs = sample_rate
        self.channels = channels
        self.exclusive = exclusive

        # Internal state
        self._device = None
        self._rec = None
        self._opened = False
        self.blocks_read = 0
        self.reconnects = 0

    def open(self) -> None:
        try:
            # Resolve device: named speaker if provided, otherwise current default
            if self.device_name:
                try:
                    device = sc.get_speaker(self.device_name)
                except Exception:
                    logger.warning(
                        f"Speaker '{self.device_name}' not found. Falling back to default."
                    )
                    device = sc.default_speaker()
            else:
                device = sc.default_speaker()

            # Create recorder with configured parameters
            rec = device.recorder(
                samplerate=self.fs,
                channels=self.channels,
                blocksize=self.blocksize,
                exclusive_mode=self.exclusive,
            )

            # Enter context to initialize internal resources
            rec.__enter__()

            # Save state
            self._device = device
            self._rec = rec
            self.device_name = device.name  # resolved, real name
            self._opened = True

            logger.info(
                f"Opened loopback on '{self.device_name}' "
                f"(fs={self.fs}, ch={self.channels}, bs={self.blocksize}, exclusive={self.exclusive})"
            )
        except Exception as e:
            logger.exception(
                f"Error opening device '{self.device_name or '[default]'}': {e}"
            )
            raise

    def close(self) -> None:
        if self._rec and self._opened:
            try:
                self._rec.__exit__(None, None, None)
            finally:
                self._opened = False
                self._rec = None
                self._device = None
                logger.info("AudioCapture closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def next_block(self, numframes: int) -> np.ndarray:
        if not self._opened or self._rec is None:
            raise RuntimeError("Recorder not opened. Call open() first.")
        if not isinstance(numframes, int) or numframes <= 0:
            raise ValueError("numframes must be a positive integer")

        try:
            block = self._rec.record(numframes=numframes)  # (frames, channels), float32 in [-1, 1]
            self.blocks_read += 1
            return block
        except Exception as e:
            # Simple reconnection: close → resolve default → reopen → retry once
            logger.warning(f"Record error ({e}). Reopening on default device...")
            self.close()
            self.device_name = None  # force default device
            self.open()
            self.reconnects += 1
            block = self._rec.record(numframes=numframes)
            self.blocks_read += 1
            return block

    def info(self) -> dict:
        return {
            "device_name": self.device_name,
            "fs": self.fs,
            "channels": self.channels,
            "blocksize": self.blocksize,
            "exclusive": self.exclusive,
            "opened": self._opened,
            "blocks_read": self.blocks_read,
            "reconnects": self.reconnects,
        }

    def capture(self, duration: float, numframes: int) -> np.ndarray:
        """
        Smoke test: capture 'duration' seconds by reading successive blocks.
        Returns a single array of shape (total_frames, channels).
        """
        if duration <= 0:
            raise ValueError("duration must be positive")

        total_frames = int(duration * self.fs)
        chunks = []
        frames_left = total_frames

        while frames_left > 0:
            nf = numframes if frames_left >= numframes else frames_left
            block = self.next_block(nf)
            chunks.append(block)
            frames_left -= block.shape[0]

        data = np.concatenate(chunks, axis=0)
        return data  # (total_frames, channels)
