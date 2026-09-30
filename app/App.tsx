/**
 * Sitewatch, on the phone.
 *
 * The server version of this posted a photo to a FastAPI box with a GPU in it. This
 * one has no network calls at all — the 10 MB graph ships inside the APK and runs on
 * the phone's own CPU. That is the entire point of the rebuild: a site with no signal
 * is exactly where a compliance check is most likely to be needed, and a detector that
 * needs a round trip is a detector that does not work in a basement.
 *
 * Two consequences shape this file. The model is loaded once at mount, because
 * creating an ORT session costs about a second and doing it per photo would dominate.
 * And the threshold slider re-runs decode and suppression over the *cached* raw
 * output rather than the network, so dragging it is free.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  LayoutChangeEvent,
  Pressable,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Asset } from 'expo-asset';
import * as ImagePicker from 'expo-image-picker';
import { useFonts } from 'expo-font';

import { Overlay } from './src/components/Overlay';
import { Threshold } from './src/components/Threshold';
import { imageToRGBA, LoadedImage } from './src/image';
import {
  DEFAULT_CONF,
  DEFAULT_IOU,
  Detection,
  Letterbox,
  decode,
  detect,
  loadModel,
  nms,
  summarise,
} from './src/detect';
import { colors, dawnArc, fonts, radius, space, type } from './src/theme';

interface Shot {
  /** Stable across re-scoring, so deleting one photo cannot disturb the others. */
  id: string;
  uri: string;
  image: LoadedImage;
  detections: Detection[];
  inferenceMs: number;
  decodeMs: number;
  /** The undecoded head, kept per photo so the slider can re-score all of them. */
  raw: Float32Array;
  anchors: number;
  box: Letterbox;
}

export default function App() {
  const [fontsLoaded] = useFonts({
    [fonts.display]: require('./assets/fonts/InstrumentSerif-Regular.ttf'),
    [fonts.body]: require('./assets/fonts/DMSans.ttf'),
  });

  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shots, setShots] = useState<Shot[]>([]);
  const [conf, setConf] = useState(DEFAULT_CONF);
  const [displayW, setDisplayW] = useState(0);
  const nextId = useRef(0);

  useEffect(() => {
    (async () => {
      try {
        // The .onnx rides along as a bundled asset; on Android it must be unpacked to
        // a real file path before ORT can open it, which downloadAsync does.
        const asset = Asset.fromModule(require('./assets/sitewatch-512.onnx'));
        await asset.downloadAsync();
        await loadModel(asset.localUri ?? asset.uri);
        setReady(true);
      } catch (e: any) {
        setError(`Could not load the model: ${e.message}`);
      }
    })();
  }, []);

  /**
   * Score one photo and append it.
   *
   * Appending rather than replacing is the point of a walk-round: a site is several
   * photographs, and the question worth answering is how many unprotected people there
   * are across all of them, not in the last one taken.
   */
  const run = useCallback(
    async (uri: string, width: number, height: number) => {
      const image = await imageToRGBA(uri, width, height);
      const result = await detect(image.small, conf, DEFAULT_IOU);
      setShots((prev) => [
        ...prev,
        {
          id: `shot-${nextId.current++}`,
          uri,
          image,
          detections: result.detections,
          inferenceMs: result.inferenceMs,
          decodeMs: image.msDecode,
          raw: result.raw,
          anchors: result.anchors,
          box: result.box,
        },
      ]);
    },
    [conf],
  );

  const pick = useCallback(
    async (fromCamera: boolean) => {
      const perm = fromCamera
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        setError(fromCamera ? 'Camera permission denied.' : 'Photo library permission denied.');
        return;
      }
      const opts: ImagePicker.ImagePickerOptions = { quality: 1, exif: false };
      const res = fromCamera
        ? await ImagePicker.launchCameraAsync(opts)
        : await ImagePicker.launchImageLibraryAsync({
            ...opts,
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsMultipleSelection: true,
          });
      if (res.canceled) return;

      setBusy(true);
      setError(null);
      try {
        // Sequentially, not in parallel: each photo holds a full RGBA buffer and an ORT
        // run, and a phone handed ten at once would run itself out of memory.
        for (const a of res.assets) {
          await run(a.uri, a.width, a.height);
        }
      } catch (e: any) {
        setError(e.message);
      } finally {
        setBusy(false);
      }
    },
    [run],
  );

  const remove = useCallback((id: string) => setShots((prev) => prev.filter((s) => s.id !== id)), []);

  /** Re-score every photo from its cached head — no second forward pass anywhere. */
  const rescore = useCallback((v: number) => {
    setConf(v);
    setShots((prev) =>
      prev.map((s) => ({
        ...s,
        detections: nms(
          decode(s.raw, s.anchors, s.box, s.image.small.width, s.image.small.height, v),
          DEFAULT_IOU,
        ),
      })),
    );
  }, []);

  // Totals across every photo still in the list, which is what a walk-round asks.
  const stats = useMemo(() => summarise(shots.flatMap((s) => s.detections)), [shots]);

  if (!fontsLoaded) return <View style={styles.boot} />;

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" />
      <LinearGradient colors={dawnArc.colors} locations={dawnArc.locations} style={styles.header}>
        <SafeAreaView edges={['top']}>
          <Text style={styles.kicker}>SITEWATCH</Text>
          <Text style={styles.display}>Hard hats,{'\n'}counted offline.</Text>
          <Text style={styles.sub}>
            YOLO11n · 2.6M parameters · running on this phone, with no network
          </Text>
        </SafeAreaView>
      </LinearGradient>

      <ScrollView style={styles.body} contentContainerStyle={styles.bodyContent}>
        <View style={styles.actions}>
          <Pressable
            style={({ pressed }) => [styles.button, styles.buttonPrimary, pressed && styles.pressed]}
            onPress={() => pick(true)}
            disabled={!ready || busy}
          >
            <Text style={styles.buttonPrimaryText}>Take a photo</Text>
          </Pressable>
          <Pressable
            style={({ pressed }) => [styles.button, pressed && styles.pressed]}
            onPress={() => pick(false)}
            disabled={!ready || busy}
          >
            <Text style={styles.buttonText}>Choose from library</Text>
          </Pressable>
        </View>

        {!ready && !error && (
          <View style={styles.notice}>
            <ActivityIndicator color={colors.onyx} />
            <Text style={styles.noticeText}>Loading the model…</Text>
          </View>
        )}
        {error && <Text style={styles.error}>{error}</Text>}

        {busy && (
          <View style={styles.notice}>
            <ActivityIndicator color={colors.onyx} />
            <Text style={styles.noticeText}>Scoring…</Text>
          </View>
        )}

        {shots.length > 0 && (
          <>
            {shots.map((shot, i) => (
              <ShotCard
                key={shot.id}
                shot={shot}
                index={i}
                total={shots.length}
                displayW={displayW}
                onLayout={(w) => setDisplayW(w)}
                onRemove={() => remove(shot.id)}
              />
            ))}

            {shots.length > 1 && <Text style={styles.totalLabel}>Across all {shots.length} photos</Text>}

            <View style={styles.statRow}>
              <Stat label="Heads found" value={String(stats.detected)} />
              <Stat label="With hard hat" value={String(stats.withHat)} />
              <Stat label="Without" value={String(stats.withoutHat)} emphasis />
            </View>

            {/* Permanent, not conditional. A warning that only appears when the app
                suspects it missed someone would be useless, because not knowing is
                exactly the failure — it cannot flag what it did not see. */}
            <Text style={styles.caveat}>
              Counts only heads the detector found. People sitting, crouching, turned away or
              partly hidden are often missed — this is not a count of everyone present, and a
              zero here is not a safe site.
            </Text>

            <Threshold value={conf} onChange={rescore} />

            <Text style={styles.timing}>
              {Math.round(shots.reduce((a, s) => a + s.inferenceMs, 0) / shots.length)} ms inference ·{' '}
              {Math.round(shots.reduce((a, s) => a + s.decodeMs, 0) / shots.length)} ms JPEG decode ·
              512×512 input{shots.length > 1 ? ', averaged' : ''}
            </Text>

            <Pressable
              style={({ pressed }) => [styles.button, pressed && styles.pressed]}
              onPress={() => setShots([])}
            >
              <Text style={styles.buttonText}>Clear all</Text>
            </Pressable>
          </>
        )}

        {shots.length === 0 && ready && !busy && (
          <Text style={styles.empty}>
            Point it at a construction site. Add as many photos as you like — the counts add
            up across all of them. Every box is drawn by a 10 MB graph inside this app;
            nothing leaves the phone.
          </Text>
        )}
      </ScrollView>
    </SafeAreaProvider>
  );
}

/**
 * One photo, its boxes, and a way to get rid of it.
 *
 * The per-photo counts sit on the card rather than only in the totals, because a
 * supervisor looking at four pictures needs to know *which* one has the violation.
 */
function ShotCard({
  shot,
  index,
  total,
  displayW,
  onLayout,
  onRemove,
}: {
  shot: Shot;
  index: number;
  total: number;
  displayW: number;
  onLayout: (w: number) => void;
  onRemove: () => void;
}) {
  // Height from the original photo, because that is the aspect ratio <Image> renders at.
  const displayH = displayW ? (displayW * shot.image.height) / shot.image.width : 0;
  const s = summarise(shot.detections);

  return (
    <View style={styles.shotCard}>
      <View style={styles.preview} onLayout={(e: LayoutChangeEvent) => onLayout(e.nativeEvent.layout.width)}>
        <Image source={{ uri: shot.uri }} style={{ width: displayW, height: displayH }} resizeMode="contain" />
        {displayW > 0 && (
          /* The viewBox is the *downscaled* image, not the original: undoing the
             letterbox leaves boxes in the coordinate space of whatever was fed to the
             network, which is the downscaled copy. Since the SVG stretches its viewBox
             to the display rect and both copies share an aspect ratio, this lines up. */
          <Overlay
            detections={shot.detections}
            imageWidth={shot.image.small.width}
            imageHeight={shot.image.small.height}
            displayWidth={displayW}
            displayHeight={displayH}
          />
        )}
      </View>
      <View style={styles.shotFoot}>
        <Text style={styles.shotMeta}>
          {total > 1 ? `${index + 1} of ${total} · ` : ''}
          {s.withHat} with hard hat
          {s.withoutHat > 0 ? ` · ${s.withoutHat} without` : ''}
        </Text>
        <Pressable
          onPress={onRemove}
          hitSlop={12}
          accessibilityLabel={`Remove photo ${index + 1}`}
          style={({ pressed }) => pressed && styles.pressed}
        >
          <Text style={styles.remove}>Remove</Text>
        </Pressable>
      </View>
    </View>
  );
}

function Stat({ label, value, emphasis }: { label: string; value: string; emphasis?: boolean }) {
  return (
    <View style={styles.stat}>
      <Text style={[styles.statValue, emphasis && styles.statValueEmphasis]}>{value}</Text>
      {/* Four cards across a phone leaves ~70pt each, which "Compliance" overflows —
          it wrapped mid-word to "Complian ce". Shrink to fit rather than wrap. */}
      <Text style={styles.statLabel} numberOfLines={1} adjustsFontSizeToFit minimumFontScale={0.75}>
        {label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  boot: { flex: 1, backgroundColor: colors.charredUmber },
  header: { paddingHorizontal: space.xl, paddingBottom: space.xxl },
  kicker: {
    ...type.caption,
    fontFamily: fonts.body,
    color: colors.parchment,
    letterSpacing: 2,
    opacity: 0.8,
    marginTop: space.lg,
  },
  display: { ...type.display, fontFamily: fonts.display, color: colors.parchment, marginTop: space.md },
  sub: { ...type.bodySm, fontFamily: fonts.body, color: colors.onyx, marginTop: space.md, opacity: 0.75 },

  body: { flex: 1, backgroundColor: colors.parchment },
  bodyContent: { padding: space.xl, gap: space.lg, paddingBottom: space.xxl * 2 },

  actions: { flexDirection: 'row', gap: space.sm },
  button: {
    flex: 1,
    borderRadius: radius.pill,
    paddingVertical: space.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.ashMist,
  },
  buttonPrimary: { backgroundColor: colors.onyx, borderColor: colors.onyx },
  buttonText: { ...type.bodySm, fontFamily: fonts.body, color: colors.onyx },
  buttonPrimaryText: { ...type.bodySm, fontFamily: fonts.body, color: colors.parchment },
  pressed: { opacity: 0.7 },

  notice: { flexDirection: 'row', alignItems: 'center', gap: space.sm },
  noticeText: { ...type.bodySm, fontFamily: fonts.body, color: colors.slateVeil },
  error: { ...type.bodySm, fontFamily: fonts.body, color: colors.onyx, backgroundColor: colors.card, padding: space.md, borderRadius: radius.small },
  empty: { ...type.body, fontFamily: fonts.body, color: colors.slateVeil, marginTop: space.lg },

  shotCard: { gap: space.sm },
  shotFoot: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: space.xs },
  shotMeta: { ...type.caption, fontFamily: fonts.body, color: colors.slateVeil, flex: 1 },
  remove: { ...type.caption, fontFamily: fonts.body, color: colors.onyx, textDecorationLine: 'underline' },
  totalLabel: { ...type.caption, fontFamily: fonts.body, color: colors.slateVeil, marginTop: space.sm },
  preview: { borderRadius: radius.card, overflow: 'hidden', backgroundColor: colors.card },
  // Written out rather than StyleSheet.absoluteFillObject, which React Native 0.86
  // no longer exposes on the type.
  busy: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(12,16,24,0.45)' },

  statRow: { flexDirection: 'row', gap: space.sm },
  stat: { flex: 1, backgroundColor: colors.card, borderRadius: radius.card, padding: space.md, gap: 2 },
  statValue: { ...type.headingSm, fontFamily: fonts.display, color: colors.onyx },
  // Emphasis by weight, never hue — the system's rule, and it survives greyscale.
  statValueEmphasis: { ...type.heading, fontFamily: fonts.display },
  statLabel: { ...type.caption, fontFamily: fonts.body, color: colors.slateVeil },

  caveat: { ...type.caption, lineHeight: 16, fontFamily: fonts.body, color: colors.slateVeil,
            backgroundColor: colors.card, borderRadius: radius.small, padding: space.md, marginTop: -space.sm },
  timing: { ...type.caption, fontFamily: fonts.body, color: colors.ashMist, textAlign: 'center' },
});
