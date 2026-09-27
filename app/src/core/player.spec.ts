import { Player } from './player';

test('synthetic silence advances in seconds from the current cursor position', async () => {
  const player = new Player();
  player.renderItems = [{ type: 'silence', absoluteStart: 1, length: 5 }];
  player.currentTime = 2;
  let clock = 10000;
  const now = jest.spyOn(Date, 'now').mockImplementation(() => clock);
  let position: (() => number) | undefined;
  jest.spyOn(player, 'playRenderItem').mockImplementation((_, getTime) => {
    position = getTime;
  });
  try {
    await player.play();
    expect(position?.()).toBe(2);
    clock += 250;
    expect(position?.()).toBe(2.25);
    clock += 750;
    expect(position?.()).toBe(3);
  } finally {
    now.mockRestore();
  }
});
