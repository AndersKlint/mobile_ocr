#include "include/mobile_ocr/mobile_ocr_plugin.h"

#include <flutter_linux/flutter_linux.h>
#include <gtk/gtk.h>

struct _MobileOcrPlugin {
  GObject parent_instance;
};

G_DEFINE_TYPE(MobileOcrPlugin, mobile_ocr_plugin, g_object_get_type())

static void mobile_ocr_plugin_init(MobileOcrPlugin* self) {}

static void mobile_ocr_plugin_dispose(GObject* object) {
  G_OBJECT_CLASS(mobile_ocr_plugin_parent_class)->dispose(object);
}

static void mobile_ocr_plugin_class_init(MobileOcrPluginClass* klass) {
  G_OBJECT_CLASS(klass)->dispose = mobile_ocr_plugin_dispose;
}

void mobile_ocr_plugin_register_with_registrar(FlPluginRegistrar* registrar) {
  g_autoptr(GObject) plugin = G_OBJECT(g_object_new(mobile_ocr_plugin_get_type(), nullptr));
}
