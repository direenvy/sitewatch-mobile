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
  uri: string;
  image: LoadedImage;
  detections: Detection[];
  inferenceMs: number;
  decodeMs: number;
}

export default function App() {
  const [fontsLoaded] = useFonts({
    [fonts.display]: require('./assets/fonts/InstrumentSerif-Regular.ttf'),
    [fonts.body]: require('./assets/fonts/DMSans.ttf'),
  });

  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shot, setShot] = useState<Shot | null>(null);
  const [conf, setConf] = useState(DEFAULT_CONF);
  const [displayW, setDisplayW] = useState(0);

  // Kept out of state on purpose: the raw head is ~12k floats and nothing re-renders
  // when it changes — only the detections derived from it do.
  const rawRef = useRef<{ raw: Float32Array; anchors: number; box: Letterbox } | null>(null);

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

  const run = useCallback(
    async (uri: string, width: number, height: number) => {
      setBusy(true);
      setError(null);
      try {
        const image = await imageToRGBA(uri, width, height);
        const result = await detect(image.small, conf, DEFAULT_IOU);
        rawRef.current = { raw: result.raw, anchors: result.anchors, box: result.box };
        setShot({
          uri,
          image,
          detections: result.detections,
          inferenceMs: result.inferenceMs,
          decodeMs: image.msDecode,
        });
      } catch (e: any) {
        setError(e.message);
      } finally {
        setBusy(false);
      }
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
        : await ImagePicker.launchImageLibraryAsync({ ...opts, mediaTypes: ImagePicker.MediaTypeOptions.Images });
      if (res.canceled) return;
      const a = res.assets[0];
      await run(a.uri, a.width, a.height);
    },
    [run],
  );

  const stats = useMemo(() => summarise(shot?.detections ?? []), [shot]);
  // Height from the original photo, because that is the aspect ratio <Image> renders
  // at. The overlay's coordinate system is a separate question — see below.
  const displayH = shot ? (displayW * shot.image.height) / shot.image.width : 0;

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

        {shot && (
          <>
            <View style={styles.preview} onLayout={(e: LayoutChangeEvent) => setDisplayW(e.nativeEvent.layout.width)}>
              <Image source={{ uri: shot.uri }} style={{ width: displayW, height: displayH }} resizeMode="contain" />
              {displayW > 0 && (
                /* The viewBox is the *downscaled* image, not the original: undoing the
                   letterbox leaves boxes in the coordinate space of whatever was fed to
                   the network, which is the downscaled copy. Since the SVG stretches its
                   viewBox to the display rect and both copies share an aspect ratio,
                   this lines up — and nothing has to be rescaled by hand. */
                <Overlay
                  detections={shot.detections}
                  imageWidth={shot.image.small.width}
                  imageHeight={shot.image.small.height}
                  displayWidth={displayW}
                  displayHeight={displayH}
                />
              )}
              {busy && (
                <View style={styles.busy}>
                  <ActivityIndicator color={colors.parchment} />
                </View>
              )}
            </View>

            <View style={styles.statRow}>
              <Stat label="People" value={String(stats.people)} />
              <Stat label="Hard hats" value={String(stats.compliant)} />
              <Stat label="Violations" value={String(stats.violations)} emphasis />
              <Stat label="Compliance" value={stats.compliancePct === null ? '—' : `${stats.compliancePct}%`} />
            </View>

            <Threshold
              value={conf}
              onChange={(v) => {
                setConf(v);
                // Re-decode from the cached head — no second forward pass.
                const cached = rawRef.current;
                if (cached && shot) {
                  const dets = nms(
                    decode(cached.raw, cached.anchors, cached.box, shot.image.small.width, shot.image.small.height, v),
                    DEFAULT_IOU,
                  );
                  setShot({ ...shot, detections: dets });
                }
              }}
            />

            <Text style={styles.timing}>
              {shot.inferenceMs} ms inference · {shot.decodeMs} ms JPEG decode · 512×512 input
            </Text>
          </>
        )}

        {!shot && ready && (
          <Text style={styles.empty}>
            Point it at a construction site. Every box is drawn by a 10 MB graph inside this
            app — nothing leaves the phone.
          </Text>
        )}
      </ScrollView>
    </SafeAreaProvider>
  );
}

function Stat({ label, value, emphasis }: { label: string; value: string; emphasis?: boolean }) {
  return (
    <View style={styles.stat}>
      <Text style={[styles.statValue, emphasis && styles.statValueEmphasis]}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
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

  preview: { borderRadius: radius.card, overflow: 'hidden', backgroundColor: colors.card },
  busy: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(12,16,24,0.45)' },

  statRow: { flexDirection: 'row', gap: space.sm },
  stat: { flex: 1, backgroundColor: colors.card, borderRadius: radius.card, padding: space.md, gap: 2 },
  statValue: { ...type.headingSm, fontFamily: fonts.display, color: colors.onyx },
  // Emphasis by weight, never hue — the system's rule, and it survives greyscale.
  statValueEmphasis: { ...type.heading, fontFamily: fonts.display },
  statLabel: { ...type.caption, fontFamily: fonts.body, color: colors.slateVeil },

  timing: { ...type.caption, fontFamily: fonts.body, color: colors.ashMist, textAlign: 'center' },
});
