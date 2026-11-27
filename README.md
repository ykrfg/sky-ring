# SKY RING

[![make.fun · Make the best flight game](https://make.fun/badge/best-flight-game-20261126-0057.svg)](https://make.fun/bounty/best-flight-game-20261126-0057)
[![SKY RING rank](https://make.fun/badge/best-flight-game-20261126-0057/ykrfg/sky-ring/rank.svg)](https://make.fun/bounty/best-flight-game-20261126-0057/leaderboard)
[![SKY RING reward](https://make.fun/badge/best-flight-game-20261126-0057/ykrfg/sky-ring/reward.svg)](https://make.fun/bounty/best-flight-game-20261126-0057)

SKY RING is an entry for the make.fun ["Make the best flight game"](https://make.fun/bounty/best-flight-game-20261126-0057) bounty.

An original 3D flight game that plays in a desktop browser, fully offline from its own files — hand-written WebGL, no game engine. Fly the FX-27 Raven from the island runway, over the sea and the coastal terrain. Two modes:

- **SORTIE** (Enter): a phased mission — take off, thread the nine numbered target rings, intercept the two bandits the tower sends you after, then come home and land. Every touchdown is graded A–D on sink rate, centreline and speed. Fuel burns with the throttle, and the tank is refilled on the runway: land, stop, and take on fuel mid-mission without losing any progress. Each sortie gets its own wind, and the in-game clock runs from afternoon through dusk into night, when the runway lights come on.
- **DOGFIGHT** (D): three waves of bandits — two, then three, then four — who manoeuvre onto your tail and shoot back with tracer fire. Hard manoeuvring spoils their aim; your hull is the score that matters.

The HUD shows altitude, speed and heading with a lit cockpit panel, an attitude indicator, the mission checklist and the tower's radio calls. Enemy aircraft are engaged by holding one in your sights until the lock tone is solid, then firing a missile.

- **Controls**: ↑/↓ pitch, ←/→ bank (bank to turn), W/S throttle, **G** landing gear, **J** or **F** fire missile, **R** recover if you get lost, **D** DOGFIGHT from the title screen, **C** credits, **O** sound settings, Esc back to title. Touch controls appear on touch devices; gentle and slow on final approach to land
- **Play locally**: `node serve.ts`, then open http://localhost:8080
- **Entry point for grading**: `node main.ts` writes the playable game, screenshots and `dossier.json` to the output directory

ブラウザでオフライン動作する3Dフライトゲームです。島の滑走路から離陸する「SORTIE」は、リング9個の通過、敵機2機の迎撃、着陸までを段階的に進めるミッションで、着陸は降下率・中心線・速度でA〜Dに採点されます。燃料はスロットルで消費し、滑走路に止まれば給油でき、進行度は保持されます。風は出撃ごとに変わり、ゲーム内時計で昼から夕方、夜へと空が移ります。「DOGFIGHT」は撃ち返してくる敵機のウェーブと戦うモードです。

## License

MIT
